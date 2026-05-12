"""Group-level policy deciding when to stop / continue / finalize aggregation.

The policy is a pure judgment layer. It reads an immutable snapshot of a
:class:`GroupAggregation` and decides what the scheduler should do next. It
does not spawn trajectories or mutate aggregator state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xtuner.v1.utils import StrEnum


if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RolloutState
    from xtuner.v1.rl.agent_loop_manager.group_aggregator import GroupAggregation


class GroupState(StrEnum):
    """Lifecycle state of a prompt-level aggregation.

    COLLECTING:
        The aggregation has fewer completed trajectories than ``min_repeat``
        and must continue to collect more.
    NEEDS_MORE:
        The aggregation has at least ``min_repeat`` completed trajectories but
        the policy determined the sampled rewards are not yet suitable for
        training (for example, all trajectories share the same reward).
        Additional trajectories should be spawned while the aggregation has
        headroom before ``max_repeat``.
    READY:
        The aggregation satisfies the training condition. The completed
        trajectories can be finalized into a group and written to the replay
        buffer.
    STOPPED:
        The aggregation reached ``max_repeat`` without producing a trainable
        group. The aggregation should be dropped; trajectories are not written
        to the replay buffer.
    """

    COLLECTING = "collecting"
    NEEDS_MORE = "needs_more"
    READY = "ready"
    STOPPED = "stopped"


class GroupPolicyConfig(BaseModel):
    """Configuration for :class:`GroupPolicy` instances.

    ``min_repeat`` is at least 2 so that algorithms that rely on within-group
    variance (for example, GRPO z-score normalization) do not degenerate.

    Args:
        min_repeat (int): Minimum number of completed trajectories before the
            aggregation can be considered for training. Must be >= 2.
        max_repeat (int): Hard upper bound on the number of trajectories that
            can be spawned for a single prompt. Must be >= ``min_repeat``.
        score_key (str): Key used to look up the scalar reward in
            ``RolloutState.reward`` when deciding whether all rewards within
            an aggregation are equal.
        score_tol (float): Floating-point tolerance used when evaluating the
            "all-equal" condition.
    """

    model_config = ConfigDict(extra="forbid")

    min_repeat: int = Field(ge=2)
    max_repeat: int = Field(ge=2)
    score_key: str = "score"
    score_tol: float = 1e-8

    @model_validator(mode="after")
    def _validate_range(self) -> "GroupPolicyConfig":
        if self.max_repeat < self.min_repeat:
            raise ValueError(
                f"max_repeat ({self.max_repeat}) must be >= min_repeat ({self.min_repeat})."
            )
        return self

    def build(self) -> "GroupPolicy":
        return DefaultGroupPolicy(self)


class GroupPolicy(ABC):
    """Pure judgment interface deciding the next action on an aggregation.

    Implementations MUST NOT mutate the aggregation passed to them. The
    aggregator guarantees the aggregation is consistent and locked while the
    policy is being evaluated.

    Args:
        config (GroupPolicyConfig): The active policy configuration. Exposed
            via ``min_repeat`` / ``max_repeat`` properties for the aggregator
            to read at prompt-registration time.
    """

    def __init__(self, config: GroupPolicyConfig) -> None:
        self._config = config

    @property
    def config(self) -> GroupPolicyConfig:
        return self._config

    @property
    def min_repeat(self) -> int:
        return self._config.min_repeat

    @property
    def max_repeat(self) -> int:
        return self._config.max_repeat

    @abstractmethod
    def on_trajectory_done(self, agg: "GroupAggregation") -> GroupState:
        """Decide the state of the aggregation after a trajectory completes.

        Args:
            agg (GroupAggregation): Snapshot of the aggregation after the
                newly-completed trajectory has been appended.

        Returns:
            GroupState: The decided state.
        """

    @abstractmethod
    def should_spawn_more(self, agg: "GroupAggregation") -> int:
        """Advise the scheduler how many more trajectories to spawn.

        Args:
            agg (GroupAggregation): Current aggregation snapshot (``completed``
                and ``in_flight`` both reflect the latest state).

        Returns:
            int: Non-negative count. Zero indicates no more trajectories are
            desirable for this prompt at this moment.
        """


class DefaultGroupPolicy(GroupPolicy):
    """All-equal-reward policy.

    Decision procedure on each completed trajectory:

    1. If ``len(completed) < min_repeat`` -> ``COLLECTING``.
    2. If the set of scores (using ``score_key``) differs beyond ``score_tol``
       -> ``READY``.
    3. Otherwise the rewards are all equal. If the aggregation still has
       headroom before ``max_repeat`` (counting both completed and
       in-flight) -> ``NEEDS_MORE``, else -> ``STOPPED``.
    """

    def on_trajectory_done(self, agg: "GroupAggregation") -> GroupState:
        completed = agg.completed
        n_completed = len(completed)
        if n_completed < self._config.min_repeat:
            return GroupState.COLLECTING

        scores = [self._extract_score(traj) for traj in completed]
        if max(scores) - min(scores) > self._config.score_tol:
            return GroupState.READY

        total = n_completed + agg.in_flight
        if total < self._config.max_repeat:
            return GroupState.NEEDS_MORE
        return GroupState.STOPPED

    def should_spawn_more(self, agg: "GroupAggregation") -> int:
        completed_and_inflight = len(agg.completed) + agg.in_flight
        # Aim for min_repeat first; once min_repeat is met, NEEDS_MORE will drive
        # further growth up to max_repeat but we still expose headroom_max here
        # so the scheduler can spawn additional trajectories when it first sees
        # a prompt (for example, warming up multi-trajectory generation).
        headroom_min = max(0, self._config.min_repeat - completed_and_inflight)
        if headroom_min > 0:
            return headroom_min
        return max(0, self._config.max_repeat - completed_and_inflight)

    def _extract_score(self, traj: "RolloutState") -> float:
        reward = getattr(traj, "reward", None)
        if reward is None:
            raise RuntimeError(
                f"Trajectory uid={getattr(traj, 'uid', None)} has no reward; "
                "judger must run before the aggregator receives the trajectory."
            )
        key = self._config.score_key
        if key not in reward:
            raise KeyError(
                f"Trajectory uid={getattr(traj, 'uid', None)} reward is missing "
                f"score_key={key!r}; reward keys were {list(reward.keys())}."
            )
        return float(reward[key])
