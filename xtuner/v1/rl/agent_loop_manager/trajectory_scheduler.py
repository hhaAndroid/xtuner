"""On-fly trajectory scheduler.

The scheduler enforces a global concurrency cap and owns a double-ended
prompt queue. Producers supply a ``runner`` coroutine that performs one
trajectory's work (deep-copy prompt, call agent loop, call judger, update
aggregator); the scheduler is responsible only for slot + queue
bookkeeping.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.utils import calculate_seq_staleness
from xtuner.v1.utils import get_logger


logger = get_logger(__name__)


ReleaseSlot = Callable[[], None]
Pipeline = Callable[["PromptRequest", ReleaseSlot], Awaitable[None]]


# Scheduler dispatch priorities. Lower = higher priority; the scheduler
# pops from the lowest-numbered non-empty bucket first. Set on
# ``PromptRequest.priority`` before ``submit`` / ``submit_front``; the
# scheduler routes the request into the bucket named by this field.
#
# Ordering rationale (highest first):
# 1. PARTIAL_RESUME — pending_keep entries from the previous round have
#    already burned thousands of tokens; resuming them recovers the most
#    inference cost.
# 2. COLLECTING — fresh sibling spawns for an aggregation that still
#    needs to reach min_repeat. Same prompt is already half-done, finish
#    it before starting new prompts.
# 3. NEEDS_MORE — re-roll for an all-equal-reward group; same prompt has
#    completed >= min_repeat trajectories, but reward variance demands
#    one more chunk. Lower than COLLECTING because the policy may still
#    decide STOPPED soon.
# 4. NEW_PROMPT — sample_prompt() output; lowest priority, fills slot
#    capacity once everything above is satisfied.
PRIORITY_PARTIAL_RESUME = 0
PRIORITY_COLLECTING = 1
PRIORITY_NEEDS_MORE = 2
PRIORITY_NEW_PROMPT = 3


@dataclass
class PromptRequest:
    """A prompt waiting for (or requesting resume of) trajectory generation.

    Args:
        prompt_uid (int): Stable per-prompt id matching
            ``aggregator.register_prompt``.
        task_name (str): Owning task.
        prompt (RolloutState): Prompt template. The scheduler never mutates
            it; the runner deep-copies before use.
        priority (int): Scheduler dispatch priority (lower = sooner). Use
            the ``PRIORITY_*`` constants. Defaults to ``PRIORITY_NEW_PROMPT``
            so prompts pulled fresh from the dataloader sit at the bottom.
    """

    prompt_uid: int
    task_name: str
    prompt: RolloutState
    priority: int = PRIORITY_NEW_PROMPT


class _PromptDeque:
    """Priority-bucketed double-ended prompt queue.

    Internally one ``deque`` per priority value. ``pop`` always drains
    the lowest-numbered non-empty bucket first; within a bucket FIFO /
    LIFO is controlled by ``push_back`` / ``push_front``. The owning
    scheduler serializes access under its lock, so this class does not
    take a lock of its own.
    """

    def __init__(self) -> None:
        self._buckets: dict[int, deque[PromptRequest]] = {}

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def __bool__(self) -> bool:
        return any(bucket for bucket in self._buckets.values())

    def push_back(self, req: PromptRequest) -> None:
        self._buckets.setdefault(req.priority, deque()).append(req)

    def push_front(self, req: PromptRequest) -> None:
        self._buckets.setdefault(req.priority, deque()).appendleft(req)

    def pop(self) -> PromptRequest:
        # Higher priority (smaller int) drains first. We sort each call;
        # the bucket count is bounded by the number of priority levels
        # (~4) so this is effectively O(1).
        for priority in sorted(self._buckets.keys()):
            bucket = self._buckets[priority]
            if bucket:
                return bucket.popleft()
        raise IndexError("pop from empty _PromptDeque")

    def lens_by_priority(self) -> dict[int, int]:
        """Snapshot of per-priority queue sizes for monitoring."""
        return {priority: len(bucket) for priority, bucket in self._buckets.items()}


def calculate_stale_threshold(max_staleness: int, sync_weights_interval: int) -> int:
    """Convert a ``max_staleness`` count into a train-step threshold.

    Matches the formula used by the legacy :class:`AsyncProduceStrategy`.

    Args:
        max_staleness (int): Number of sync intervals a sample may lag.
        sync_weights_interval (int): How often the trainer syncs weights.

    Returns:
        int: Threshold in train-step units.
    """
    if max_staleness < 0:
        raise ValueError(f"max_staleness must be non-negative, got {max_staleness}.")
    if sync_weights_interval <= 0:
        raise ValueError(f"sync_weights_interval must be positive, got {sync_weights_interval}.")
    return (max_staleness + 1) * sync_weights_interval


class TrajectorySchedulerConfig(BaseModel):
    """Configuration for :class:`TrajectoryScheduler`.

    Args:
        max_on_fly (int): Global concurrent trajectory cap. Must be > 0.
        prompt_preload_count (int): Hint to the producer about how many
            fresh prompts should sit in the queue at any moment. Replaces
            the legacy ``over_sample_threshold`` knob.
        wait_until_all_ready (bool): If True, ``produce_batch`` drains the
            scheduler before returning. Mirrors the old
            ``SyncProduceStrategy`` behavior for colocated training.
        max_staleness (int): Tolerance for ``seq_staleness`` before a
            trajectory is considered model-expired.
        enable_partial_rollout (bool): When True, ABORTED trajectories are
            pushed into ``GroupAggregation.pending_keep`` and resumed on
            the next spawn for the same prompt.
        tail_batch_trigger_size (int): Threshold of pending-keep
            trajectories that flips the scheduler into tail-batch mode.
        cleanup_timeout_s (float): Upper bound on
            :meth:`TrajectoryScheduler.pause_and_cleanup` before outstanding
            tasks are force-cancelled.
    """

    model_config = ConfigDict(extra="forbid")

    max_on_fly: int = Field(gt=0)
    prompt_preload_count: int = Field(ge=0, default=0)
    wait_until_all_ready: bool = False
    max_staleness: int = Field(ge=0, default=0)
    enable_partial_rollout: bool = False
    tail_batch_trigger_size: int = Field(ge=0, default=0)
    cleanup_timeout_s: float = Field(gt=0.0, default=300.0)

    @model_validator(mode="after")
    def _validate_partial_and_staleness(self) -> "TrajectorySchedulerConfig":
        if not self.enable_partial_rollout and self.max_staleness > 0:
            logger.warning(
                "TrajectorySchedulerConfig: enable_partial_rollout is False but "
                "max_staleness > 0; consider enabling partial rollout to reuse "
                "ABORTED trajectory tokens across weight syncs."
            )
        return self

    def build(self, *, sync_weights_interval: int) -> "TrajectoryScheduler":
        return TrajectoryScheduler(self, sync_weights_interval=sync_weights_interval)


class TrajectoryScheduler:
    """Bounded-concurrency trajectory scheduler.

    Pipelines have two phases. The inference phase consumes a slot of the
    ``max_on_fly`` budget (one slot per trajectory currently calling the
    rollout engine). The post phase, which runs partial-rollout merging,
    aggregator decisions, and replay-buffer writes, does not consume an
    inference slot but is still tracked so :meth:`drain` /
    :meth:`pause_and_cleanup` can wait for it before checkpointing.

    A pipeline is a coroutine ``pipeline(req, release_slot)``. It must call
    ``release_slot()`` once the inference phase is done so the scheduler can
    spawn a new inference. If the pipeline returns without calling it, the
    wrapper releases the slot for it on exit.

    The caller is expected to stop submitting before :meth:`drain` or
    :meth:`pause_and_cleanup` to avoid starving the wait loop.

    Args:
        config (TrajectorySchedulerConfig): Scheduler configuration.
        sync_weights_interval (int): Train-step interval between weight
            syncs, combined with ``config.max_staleness`` to derive the
            expiration threshold.
    """

    def __init__(
        self, config: TrajectorySchedulerConfig, *, sync_weights_interval: int
    ) -> None:
        self._config = config
        self._queue = _PromptDeque()
        # All in-flight pipeline tasks (inference + post). Used for drain /
        # pause_and_cleanup. Includes both phases so checkpointing waits for
        # post-phase replay-buffer writes to finish.
        self._tasks: set[asyncio.Task] = set()
        # Number of pipelines currently in their inference phase. Strictly
        # bounded by ``max_on_fly``; this is the only counter that
        # constrains :meth:`spawn_if_slot`.
        self._inflight = 0
        self._lock = asyncio.Lock()
        # Edge-triggered "something released or completed" signal. Bumped by
        # ``release_slot`` and ``_on_task_done``; consumed by
        # :meth:`wait_first_completed`.
        self._progress = asyncio.Event()
        self._stale_threshold = calculate_stale_threshold(
            config.max_staleness, sync_weights_interval
        )

    @property
    def config(self) -> TrajectorySchedulerConfig:
        return self._config

    @property
    def stale_threshold(self) -> int:
        return self._stale_threshold

    def pending_count(self) -> int:
        """Total in-flight pipeline tasks (inference + post phase)."""
        return len(self._tasks)

    def inflight_count(self) -> int:
        """Pipelines currently in their inference phase. Bounded by ``max_on_fly``."""
        return self._inflight

    def queue_len(self) -> int:
        """Number of prompts waiting in the queue."""
        return len(self._queue)

    def queue_lens_by_priority(self) -> dict[int, int]:
        """Per-priority queue sizes; keys are ``PRIORITY_*`` constants."""
        return self._queue.lens_by_priority()

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        """True when ``seq_staleness(model_step, train_step) >= threshold``."""
        return calculate_seq_staleness(model_step, train_step) >= self._stale_threshold

    async def submit(self, req: PromptRequest) -> None:
        """Enqueue a prompt at the back (default for fresh prompts)."""
        async with self._lock:
            self._queue.push_back(req)

    async def submit_front(self, req: PromptRequest) -> None:
        """Enqueue a prompt at the front.

        Used for NEEDS_MORE re-entries and prompts whose aggregation has
        pending-keep trajectories awaiting resume; the goal is to minimize
        the model-step gap between trajectories of the same group.
        """
        async with self._lock:
            self._queue.push_front(req)

    async def spawn_if_slot(self, pipeline: Pipeline) -> bool:
        """Try to spawn one pipeline if an inference slot and a prompt are available.

        Non-blocking: returns ``False`` when the inference phase is at
        ``max_on_fly`` or the queue is empty. Otherwise creates an
        ``asyncio.Task`` running ``pipeline(req, release_slot)`` and tracks
        it; ``release_slot`` is a synchronous callback the pipeline must
        invoke once its inference phase is done so the slot can be reused
        while the post phase is still running.

        Args:
            pipeline (Pipeline): Coroutine factory ``(req, release_slot)``
                covering inference plus post processing.

        Returns:
            bool: Whether a pipeline was spawned.
        """
        async with self._lock:
            if self._inflight >= self._config.max_on_fly:
                return False
            if len(self._queue) == 0:
                return False
            req = self._queue.pop()
            self._inflight += 1

        released = False

        def release_slot() -> None:
            # Idempotent so the wrapper's safety-net call is harmless.
            nonlocal released
            if released:
                return
            released = True
            self._inflight -= 1
            self._progress.set()

        async def wrapper() -> None:
            try:
                await pipeline(req, release_slot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - must not break scheduler state
                logger.error(
                    f"Pipeline failed for prompt_uid={req.prompt_uid}: "
                    f"{type(exc).__name__}: {exc}",
                    exc_info=exc,
                )
            finally:
                if not released:
                    release_slot()

        task = asyncio.create_task(wrapper())
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return True

    async def drain(self) -> None:
        """Wait for all pipelines (inference + post phase) to finish.

        Called from the colocated path (``wait_until_all_ready=True``) after
        the producer stops spawning. Does not cancel tasks.
        """
        while True:
            async with self._lock:
                pending = set(self._tasks)
            if not pending:
                return
            await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)

    async def wait_first_completed(self, timeout_s: float | None = None) -> None:
        """Wait until an inference slot is released or a pipeline finishes.

        Used by the main produce loop to yield control while pending work
        makes progress. Returns immediately when nothing is in flight.
        """
        if self._inflight == 0 and not self._tasks:
            return
        self._progress.clear()
        try:
            await asyncio.wait_for(self._progress.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return

    async def pause_and_cleanup(self) -> float:
        """Drain pipelines; cancel any still running after the timeout.

        The caller is expected to abort underlying rollout generation before
        calling this so inference-phase tasks can terminate quickly. Post-
        phase work is always allowed to drain so completed trajectories
        land in the replay buffer before checkpointing.

        Returns:
            float: Elapsed time in seconds.
        """
        start = time.perf_counter()
        timeout = self._config.cleanup_timeout_s
        while True:
            async with self._lock:
                pending = set(self._tasks)
            if not pending:
                break
            elapsed = time.perf_counter() - start
            if elapsed >= timeout:
                await self._cancel_all(pending)
                break
            remaining = max(0.1, timeout - elapsed)
            await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.ALL_COMPLETED
            )
        return time.perf_counter() - start

    async def clear_queue(self) -> list[PromptRequest]:
        """Drain and return all queued prompts.

        Used by the producer / manager when preparing a checkpoint: queued
        prompts that have not yet been dispatched are captured so they can
        be re-submitted on resume.
        """
        async with self._lock:
            drained: list[PromptRequest] = []
            while self._queue:
                drained.append(self._queue.pop())
            return drained

    def _on_task_done(self, task: asyncio.Task) -> None:
        # done callback runs synchronously; set.discard is GIL-atomic
        # for a single element, so no lock is required here.
        self._tasks.discard(task)
        self._progress.set()

    async def _cancel_all(self, tasks: set[asyncio.Task]) -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.warning(
            f"TrajectoryScheduler: cancelled {len(tasks)} trajectory tasks after cleanup timeout."
        )


_PRIORITY_LABEL = {
    PRIORITY_PARTIAL_RESUME: "resume",
    PRIORITY_COLLECTING: "coll",
    PRIORITY_NEEDS_MORE: "more",
    PRIORITY_NEW_PROMPT: "new",
}


def format_queue_breakdown(lens_by_priority: dict[int, int]) -> str:
    """Render priority->size dict as ``resume:N/coll:N/more:N/new:N``.

    Always prints all four canonical buckets (zero-fill) so the column
    width stays stable across log lines; unknown priorities are appended
    as ``p<N>:M``. Used by the producer's per-iter and per-batch logs to
    show how requests pile up by priority class.
    """
    canonical = [
        PRIORITY_PARTIAL_RESUME,
        PRIORITY_COLLECTING,
        PRIORITY_NEEDS_MORE,
        PRIORITY_NEW_PROMPT,
    ]
    parts = [f"{_PRIORITY_LABEL[p]}:{lens_by_priority.get(p, 0)}" for p in canonical]
    extras = sorted(p for p in lens_by_priority if p not in _PRIORITY_LABEL)
    parts.extend(f"p{p}:{lens_by_priority[p]}" for p in extras)
    return "/".join(parts)
