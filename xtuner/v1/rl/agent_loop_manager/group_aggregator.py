"""Per-prompt trajectory aggregator.

The aggregator owns the in-progress state of every prompt currently being
sampled. A :class:`GroupAggregation` tracks ``completed`` trajectories,
``pending_keep`` trajectories held for partial-rollout resume, and an
``in_flight`` counter updated by the scheduler under a strict ownership
protocol.

:class:`GroupAggregator` serializes mutations behind a single
``asyncio.Lock`` and guarantees that :meth:`GroupAggregator.add_trajectory`
performs "append completed -> policy judgment -> finalize-if-READY"
atomically.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace

from xtuner.v1.data_proto.rl_data import RolloutState, refresh_seq_staleness
from xtuner.v1.rl.agent_loop_manager.group_policy import GroupPolicy, GroupState
from xtuner.v1.rl.utils import free_rollout_state_list_refs
from xtuner.v1.utils import get_logger


logger = get_logger(__name__)


@dataclass
class GroupAggregation:
    """In-memory state of trajectories for a single prompt.

    ``completed`` holds ``Status.COMPLETED`` trajectories after judger
    scoring. ``pending_keep`` holds ABORTED trajectories preserved for
    partial-rollout resume.

    The aggregator mutates the aggregation under its internal lock; callers
    must not mutate returned aggregations directly.

    Args:
        prompt_uid (int): Stable per-prompt id, typically
            ``RolloutState.message_uid`` assigned by the sampler.
        task_name (str): Owning task name.
        original_prompt (RolloutState): Prompt template used by the scheduler
            to spawn new trajectories. The scheduler performs
            ``copy.deepcopy`` before each spawn, so the template is not
            mutated.
        min_repeat (int): Minimum completed count before the aggregation can
            become READY.
        max_repeat (int): Hard upper bound on total spawned trajectories for
            this prompt.
        completed (list[RolloutState]): Completed trajectories appended in
            arrival order. Emptied on finalize.
        pending_keep (list[RolloutState]): ABORTED trajectories reserved for
            partial-rollout resume. Emptied on drop / finalize.
        in_flight (int): Number of trajectories spawned but not yet observed
            by :meth:`GroupAggregator.add_trajectory`.
        created_ts (float): ``time.monotonic()`` at registration, for TTL
            diagnostics.
    """

    prompt_uid: int
    task_name: str
    original_prompt: RolloutState
    min_repeat: int
    max_repeat: int
    completed: list[RolloutState] = field(default_factory=list)
    pending_keep: list[RolloutState] = field(default_factory=list)
    in_flight: int = 0
    created_ts: float = field(default_factory=time.monotonic)
    is_batch_judger: bool = False


class GroupAggregator:
    """Per-prompt trajectory aggregator.

    Concurrency contract:

    * All public methods acquire an internal ``asyncio.Lock``.
    * :meth:`add_trajectory` atomically appends a completed trajectory,
      evaluates the policy, and, when the policy returns ``READY``, pops
      the completed list. Callers either receive a ready-to-write group or
      a signal to submit the prompt back to the queue; no external
      synchronization is required.
    * In-flight accounting follows a strict ownership protocol: the
      scheduler calls :meth:`mark_in_flight` with ``+1`` before spawning a
      trajectory and with ``-1`` in a ``finally`` block regardless of
      outcome. :meth:`add_trajectory` does NOT decrement ``in_flight``.

    Args:
        policy (GroupPolicy): Judgment policy used to decide
            COLLECTING / NEEDS_MORE / READY / STOPPED. Its ``min_repeat`` and
            ``max_repeat`` seed new aggregations at registration time.
    """

    def __init__(self, policy: GroupPolicy) -> None:
        self._policy = policy
        self._groups: dict[int, GroupAggregation] = {}
        # Cached count of trajectories sitting in any aggregation's
        # ``completed`` list. Used by the producer (Q3) to fold partial-
        # progress into ``available_for_unit`` in O(1) without walking every
        # aggregation each iter. Maintained by add_trajectory / drop / clear /
        # refresh_aggregation_staleness / try_finalize_if_ready /
        # load_state_dict; ``register_prompt`` does not bump it because new
        # aggregations start with empty ``completed``.
        self._completed_trajectory_count = 0
        self._lock = asyncio.Lock()

    @property
    def policy(self) -> GroupPolicy:
        return self._policy

    async def register_prompt(
        self,
        prompt: RolloutState,
        task_name: str,
        is_batch_judger: bool = False,
    ) -> GroupAggregation:
        """Create a new aggregation keyed by ``prompt.message_uid``.

        Args:
            prompt (RolloutState): Prompt template. The aggregator stores a
                reference; the scheduler deep-copies it at each spawn, so
                callers must not mutate it after this call.
            task_name (str): Task owning this prompt.
            is_batch_judger (bool): When True the aggregation's judger
                expects to receive the whole completed group at once, so
                trajectories arriving via :meth:`add_trajectory` will not
                yet have ``reward`` populated. The policy short-circuits to
                ``READY`` once ``len(completed) >= min_repeat``.

        Returns:
            GroupAggregation: The newly registered aggregation.
        """
        if prompt.message_uid is None:
            raise ValueError("Prompt must have message_uid set before registration.")
        async with self._lock:
            prompt_uid = prompt.message_uid
            if prompt_uid in self._groups:
                raise KeyError(
                    f"Aggregation for prompt_uid={prompt_uid} already exists; "
                    "each prompt must only be registered once."
                )
            agg = GroupAggregation(
                prompt_uid=prompt_uid,
                task_name=task_name,
                original_prompt=prompt,
                min_repeat=self._policy.min_repeat,
                max_repeat=self._policy.max_repeat,
                is_batch_judger=is_batch_judger,
            )
            self._groups[prompt_uid] = agg
            return agg

    async def add_trajectory(
        self, traj: RolloutState
    ) -> tuple[GroupState | None, list[RolloutState] | None]:
        """Record a completed trajectory and consult the policy.

        The judger must have assigned ``traj.reward`` before this call.

        Args:
            traj (RolloutState): The ``Status.COMPLETED`` trajectory. Its
                ``message_uid`` identifies the aggregation to update.

        Returns:
            tuple[GroupState | None, list[RolloutState] | None]: Two cases.
            ``(state, finalized)`` when the aggregation is known; the second
            element is non-``None`` iff ``state is GroupState.READY``.
            ``(None, None)`` when the trajectory is orphaned (its aggregation
            was already finalized or dropped, e.g. a late in-flight
            completion). The caller should discard orphaned trajectories.
        """
        if traj.message_uid is None:
            raise ValueError(
                f"Trajectory uid={traj.uid} has no message_uid; cannot route to aggregation."
            )
        async with self._lock:
            prompt_uid = traj.message_uid
            agg = self._groups.get(prompt_uid)
            if agg is None:
                logger.debug(
                    f"Trajectory uid={traj.uid} arrived after aggregation "
                    f"prompt_uid={prompt_uid} was already finalized/dropped; discarding."
                )
                return None, None
            agg.completed.append(traj)
            self._completed_trajectory_count += 1
            state = self._policy.on_trajectory_done(agg)
            if state is GroupState.READY:
                finalized = agg.completed
                # Remove the aggregation atomically so late in-flight
                # trajectories for this prompt are treated as orphans.
                self._groups.pop(prompt_uid)
                # The finalized group is leaving the aggregator and entering
                # the replay buffer; its trajectories should no longer count
                # toward the producer's "in-flight progress" view.
                self._completed_trajectory_count -= len(finalized)
                return state, finalized
            return state, None

    async def push_pending_keep(self, traj: RolloutState) -> None:
        """Record an ABORTED trajectory for later partial-rollout resume.

        Silently discards the trajectory when no aggregation exists (the
        prompt has already been finalized or dropped).

        Args:
            traj (RolloutState): Aborted trajectory whose ``message_uid``
                identifies the aggregation.
        """
        if traj.message_uid is None:
            raise ValueError(
                f"Trajectory uid={traj.uid} has no message_uid; cannot route to aggregation."
            )
        async with self._lock:
            agg = self._groups.get(traj.message_uid)
            if agg is None:
                return
            agg.pending_keep.append(traj)

    async def pop_pending_keep(self, prompt_uid: int) -> RolloutState | None:
        """Pop the oldest pending-keep trajectory for resume.

        Returns ``None`` when the aggregation is missing or ``pending_keep``
        is empty. The caller feeds the returned trajectory into the
        partial-rollout resume path.
        """
        async with self._lock:
            agg = self._groups.get(prompt_uid)
            if agg is None or not agg.pending_keep:
                return None
            return agg.pending_keep.pop(0)

    async def has_pending_keep(self, prompt_uid: int) -> bool:
        async with self._lock:
            agg = self._groups.get(prompt_uid)
            return agg is not None and bool(agg.pending_keep)

    async def mark_in_flight(self, prompt_uid: int, delta: int) -> None:
        """Adjust the in-flight count of an aggregation.

        Args:
            prompt_uid (int): Aggregation identifier.
            delta (int): ``+1`` before spawning; ``-1`` after the spawning
                coroutine exits, including cancel and exception paths.
        """
        async with self._lock:
            agg = self._groups.get(prompt_uid)
            if agg is None:
                # The aggregation may have been finalized or dropped
                # concurrently with a -1 decrement; treat as a no-op.
                return
            agg.in_flight += delta
            if agg.in_flight < 0:
                raise RuntimeError(
                    f"in_flight went negative for prompt_uid={prompt_uid}; "
                    "mark_in_flight was decremented more times than incremented."
                )

    async def drop(self, prompt_uid: int) -> None:
        """Remove an aggregation unconditionally, for example on STOPPED.

        Frees plasma ObjectRefs (``routed_experts`` / ``mm_info`` pixel
        values) on every completed and pending-keep trajectory inside the
        aggregation before letting Python GC reclaim the rest. Distributed
        ref counting in Ray is unreliable for tensors that have travelled
        across actor boundaries, so dropping silently would otherwise leak
        the per-token routing tensor into plasma until process exit.
        """
        async with self._lock:
            agg = self._groups.pop(prompt_uid, None)
            if agg is not None:
                self._completed_trajectory_count -= len(agg.completed)
        if agg is None:
            return
        # Released outside the lock so plasma free does not serialise with
        # other aggregator mutations.
        free_rollout_state_list_refs(list(agg.completed) + list(agg.pending_keep))

    async def clear(self) -> int:
        """Drop every aggregation and free their trajectory ObjectRefs.

        Used by the producer's pause_produce path: when a colocated
        produce_batch returns target_met (or pause_and_cleanup cancels
        in-flight trajectories ahead of a weight sync), every aggregation
        still in COLLECTING / NEEDS_MORE without a finalising trajectory
        on the way becomes a zombie — no future trajectory will arrive to
        flip it to READY or STOPPED, so it would otherwise sit in
        ``_groups`` forever. Each carries up to ``min_repeat - 1``
        completed trajectories with their ``routed_experts`` ObjectRefs
        pinned in plasma; clearing them out keeps active_aggregations
        bounded across train steps and stops the producer's per-iter lock
        traffic from growing without bound.

        Returns the number of aggregations removed.
        """
        async with self._lock:
            groups = list(self._groups.values())
            self._groups.clear()
            self._completed_trajectory_count = 0
        if not groups:
            return 0
        all_trajs: list[RolloutState] = []
        for agg in groups:
            all_trajs.extend(agg.completed)
            all_trajs.extend(agg.pending_keep)
        free_rollout_state_list_refs(all_trajs)
        return len(groups)

    async def exists(self, prompt_uid: int) -> bool:
        async with self._lock:
            return prompt_uid in self._groups

    async def get_snapshot(self, prompt_uid: int) -> GroupAggregation | None:
        """Return a shallow copy of an aggregation for read-only inspection."""
        async with self._lock:
            agg = self._groups.get(prompt_uid)
            if agg is None:
                return None
            return replace(
                agg,
                completed=list(agg.completed),
                pending_keep=list(agg.pending_keep),
            )

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._groups)

    async def completed_trajectory_count(self) -> int:
        """O(1) total of trajectories sitting in any aggregation's ``completed`` list.

        The producer folds this into ``available_for_unit`` so an
        in-progress trajectory that has already finished generating but
        not yet finalised into a group counts toward ``target_abs``. This
        avoids over-sampling fresh prompts when most of the work for the
        next batch is already done — the same prompt's other trajectories
        will eventually finalise the group and write it to the buffer.
        """
        async with self._lock:
            return self._completed_trajectory_count

    async def list_unfinished_snapshots(self) -> list[GroupAggregation]:
        """Return shallow snapshots of every active aggregation.

        Used by the producer at the start of each colocated produce_batch
        to resume aggregations whose in-flight trajectories were cancelled
        by the previous step's pause_produce. Each returned snapshot owns
        its own ``completed`` / ``pending_keep`` lists, so the caller can
        iterate without holding the aggregator lock.
        """
        async with self._lock:
            return [
                replace(
                    agg,
                    completed=list(agg.completed),
                    pending_keep=list(agg.pending_keep),
                )
                for agg in self._groups.values()
            ]

    async def refresh_aggregation_staleness(
        self,
        prompt_uid: int,
        current_train_step: int,
        stale_threshold: int,
    ) -> tuple[int, int]:
        """Drop stale trajectories from a single aggregation.

        Recomputes ``seq_staleness`` for every trajectory in ``completed``
        and ``pending_keep`` against ``current_train_step``, then removes
        those whose staleness has reached ``stale_threshold``. Removed
        trajectories' ``routed_experts`` ObjectRefs are freed.

        The aggregation itself is **not** dropped even if both lists end
        up empty — the original prompt is preserved so the producer can
        re-spawn fresh trajectories under the same prompt_uid.

        Args:
            prompt_uid (int): Aggregation identifier.
            current_train_step (int): Train step to recompute staleness against.
            stale_threshold (int): Inclusive cap on per-token staleness; a
                trajectory at or above this value is dropped.

        Returns:
            tuple[int, int]: ``(stale_completed, stale_pending_keep)`` —
            the number of trajectories removed from each list.
        """
        if stale_threshold <= 0:
            raise ValueError(f"stale_threshold must be positive, got {stale_threshold}.")
        async with self._lock:
            agg = self._groups.get(prompt_uid)
            if agg is None:
                return 0, 0
            stale_completed: list[RolloutState] = []
            kept_completed: list[RolloutState] = []
            refresh_seq_staleness(agg.completed, current_train_step)
            for traj in agg.completed:
                if getattr(traj, "seq_staleness", 0) >= stale_threshold:
                    stale_completed.append(traj)
                else:
                    kept_completed.append(traj)
            stale_pending: list[RolloutState] = []
            kept_pending: list[RolloutState] = []
            refresh_seq_staleness(agg.pending_keep, current_train_step)
            for traj in agg.pending_keep:
                if getattr(traj, "seq_staleness", 0) >= stale_threshold:
                    stale_pending.append(traj)
                else:
                    kept_pending.append(traj)
            agg.completed = kept_completed
            agg.pending_keep = kept_pending
            self._completed_trajectory_count -= len(stale_completed)
        # Free outside the lock to avoid stalling other aggregator mutations.
        if stale_completed or stale_pending:
            free_rollout_state_list_refs(stale_completed + stale_pending)
        return len(stale_completed), len(stale_pending)

    async def try_finalize_if_ready(self, prompt_uid: int) -> list[RolloutState] | None:
        """Re-evaluate the policy on an aggregation; finalise if READY.

        Used by the producer's cross-step resume path: after staleness
        refresh, an aggregation may already satisfy the policy
        (``completed >= min_repeat`` with non-zero reward variance). In
        that case there's no point spawning more trajectories — emit the
        group now so it lands in the replay buffer.

        STOPPED is *not* handled here because the producer's resume path
        decides separately whether to drop or re-spawn; this method's
        contract is "if it can become a buffer-ready group right now,
        give it to me, otherwise leave the aggregation alone."

        Returns:
            list[RolloutState] | None: The finalised group, or ``None``
            when the aggregation is missing or not yet READY.
        """
        async with self._lock:
            agg = self._groups.get(prompt_uid)
            if agg is None or not agg.completed:
                return None
            state = self._policy.on_trajectory_done(agg)
            if state is not GroupState.READY:
                return None
            finalized = agg.completed
            self._groups.pop(prompt_uid)
            self._completed_trajectory_count -= len(finalized)
            return finalized

    async def state_dict(self) -> dict:
        """Serialize aggregator state for checkpointing.

        Returns ``completed`` and ``pending_keep`` trajectories plus
        per-prompt metadata. ``in_flight`` is NOT preserved: on resume it is
        reset to zero and any outstanding trajectory tasks are expected to
        have been cancelled or drained by the caller.
        """
        async with self._lock:
            return {
                "groups": [
                    {
                        "prompt_uid": agg.prompt_uid,
                        "task_name": agg.task_name,
                        "original_prompt": agg.original_prompt.model_dump(),
                        "min_repeat": agg.min_repeat,
                        "max_repeat": agg.max_repeat,
                        "completed": [item.model_dump() for item in agg.completed],
                        "pending_keep": [item.model_dump() for item in agg.pending_keep],
                        "created_ts": agg.created_ts,
                        "is_batch_judger": agg.is_batch_judger,
                    }
                    for agg in self._groups.values()
                ],
            }

    async def load_state_dict(self, state: dict) -> None:
        """Restore aggregator state. Existing state is replaced."""
        async with self._lock:
            self._groups.clear()
            self._completed_trajectory_count = 0
            for entry in state.get("groups", []):
                original_prompt = RolloutState.model_validate(entry["original_prompt"])
                agg = GroupAggregation(
                    prompt_uid=entry["prompt_uid"],
                    task_name=entry["task_name"],
                    original_prompt=original_prompt,
                    min_repeat=entry["min_repeat"],
                    max_repeat=entry["max_repeat"],
                    completed=[RolloutState.model_validate(item) for item in entry["completed"]],
                    pending_keep=[
                        RolloutState.model_validate(item) for item in entry["pending_keep"]
                    ],
                    in_flight=0,
                    created_ts=entry.get("created_ts", time.monotonic()),
                    is_batch_judger=entry.get("is_batch_judger", False),
                )
                self._groups[agg.prompt_uid] = agg
                self._completed_trajectory_count += len(agg.completed)
