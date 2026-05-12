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


Runner = Callable[["PromptRequest"], Awaitable[None]]


@dataclass
class PromptRequest:
    """A prompt waiting for (or requesting resume of) trajectory generation.

    Args:
        prompt_uid (int): Stable per-prompt id matching
            ``aggregator.register_prompt``.
        task_name (str): Owning task.
        prompt (RolloutState): Prompt template. The scheduler never mutates
            it; the runner deep-copies before use.
        priority (int): Informational. The scheduler relies on queue position
            (front vs back) rather than this field for ordering; preserved
            for logging and debugging.
    """

    prompt_uid: int
    task_name: str
    prompt: RolloutState
    priority: int = 0


class _PromptDeque:
    """Non-thread-safe double-ended prompt queue.

    The owning scheduler serializes access under its lock, so this class
    does not take a lock of its own.
    """

    def __init__(self) -> None:
        self._buf: deque[PromptRequest] = deque()

    def __len__(self) -> int:
        return len(self._buf)

    def __bool__(self) -> bool:
        return bool(self._buf)

    def push_back(self, req: PromptRequest) -> None:
        self._buf.append(req)

    def push_front(self, req: PromptRequest) -> None:
        self._buf.appendleft(req)

    def pop(self) -> PromptRequest:
        return self._buf.popleft()


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

    The scheduler owns three pieces of state behind a single
    ``asyncio.Lock``: the prompt queue, the pending-task set, and
    configuration. Business logic (deep-copy prompt, call agent loop, call
    judger, update aggregator) lives in the runner callable passed to
    :meth:`spawn_if_slot`.

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
        self._pending: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
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
        """Current number of in-flight trajectory tasks."""
        return len(self._pending)

    def queue_len(self) -> int:
        """Number of prompts waiting in the queue."""
        return len(self._queue)

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

    async def spawn_if_slot(self, runner: Runner) -> bool:
        """Try to spawn one trajectory if a slot and a prompt are available.

        Non-blocking: returns ``False`` when the pending set is full or the
        queue is empty. Otherwise creates an ``asyncio.Task`` running
        ``runner(req)`` and tracks it in the pending set.

        Args:
            runner (Runner): Coroutine factory that performs one trajectory's
                work given a :class:`PromptRequest`.

        Returns:
            bool: Whether a trajectory was spawned.
        """
        async with self._lock:
            if len(self._pending) >= self._config.max_on_fly:
                return False
            if len(self._queue) == 0:
                return False
            req = self._queue.pop()
            task = asyncio.create_task(runner(req))
            self._pending.add(task)
            task.add_done_callback(self._on_task_done)
            return True

    async def drain(self) -> None:
        """Wait for all pending trajectories to finish.

        Called from the colocated path (``wait_until_all_ready=True``) after
        the producer stops spawning. Does not cancel tasks.
        """
        while True:
            async with self._lock:
                pending = set(self._pending)
            if not pending:
                return
            await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)

    async def pause_and_cleanup(self) -> float:
        """Drain pending tasks; cancel any still running after the timeout.

        The caller is expected to abort underlying rollout generation before
        calling this so pending tasks can terminate quickly.

        Returns:
            float: Elapsed time in seconds.
        """
        start = time.perf_counter()
        timeout = self._config.cleanup_timeout_s
        while True:
            async with self._lock:
                pending = set(self._pending)
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
        self._pending.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"Trajectory task failed: {type(exc).__name__}: {exc}",
                exc_info=exc,
            )

    async def _cancel_all(self, tasks: set[asyncio.Task]) -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.warning(
            f"TrajectoryScheduler: cancelled {len(tasks)} trajectory tasks after cleanup timeout."
        )
