import asyncio
import copy
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable
from uuid import uuid4

import ray
import tqdm
from mmengine.dist import get_rank
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    Status,
    get_group_status,
)
from xtuner.v1.rl.agent_loop import AgentLoopSpec, get_agent_loop_rollout_ctl
from xtuner.v1.rl.agent_loop_manager.group_aggregator import GroupAggregator
from xtuner.v1.rl.agent_loop_manager.group_policy import (
    GroupPolicy,
    GroupPolicyConfig,
    GroupState,
)
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.agent_loop_manager.trajectory_scheduler import (
    PRIORITY_COLLECTING,
    PRIORITY_NEEDS_MORE,
    PRIORITY_NEW_PROMPT,
    PRIORITY_PARTIAL_RESUME,
    Pipeline,
    PromptRequest,
    TrajectoryScheduler,
    TrajectorySchedulerConfig,
    format_queue_breakdown,
)
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout.utils import pause_generation
from xtuner.v1.rl.utils import calculate_seq_staleness, create_task, free_rollout_state_refs
from xtuner.v1.utils import get_logger

from .sampler import Sampler, SamplerExhausted


logger = get_logger()
GROUP_GENERATE_TIME_KEY = "group_generate_time_s"
# Minimum seconds between per-iteration progress logs inside produce_batch's
# main loop. Throttled purely by wall-clock so log volume stays bounded
# regardless of how fast ``available`` changes.
PRODUCE_PROGRESS_LOG_INTERVAL_S = 30.0


class _ProgressDisplayer:
    def __init__(self, progress_bar: Any | None) -> None:
        self._tqdm = progress_bar

    @classmethod
    def create(cls, *, strategy_name: str, task_name: str, total: int, initial: int) -> "_ProgressDisplayer":
        total = max(0, total)
        initial = min(total, max(0, initial))
        if total <= 0 or get_rank() != 0:
            return cls(None)
        return cls(
            tqdm.tqdm(
                total=total,
                initial=initial,
                desc=f"{strategy_name} {task_name}",
                unit="sample",
                dynamic_ncols=True,
                mininterval=30,
                leave=False,
            )
        )

    def update(self, value: int) -> None:
        if self._tqdm is None:
            return
        total = max(0, int(self._tqdm.total or 0))
        value = min(total, max(0, value))
        delta = value - self._tqdm.n
        if delta > 0:
            self._tqdm.update(delta)
            self._tqdm.n = value

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None


@dataclass
class ProduceProgress:
    """生产者和消费者共享的 live 进度对象。

    设计目标：
    - Manager / 调用方负责初始化并原地更新这个对象，strategy 只接收引用并读取最新进度。
    - target / consumed 使用全局绝对累计口径，避免 consumer 取走 buffer 中的 completed 后，
      producer 把已消费样本误判成缺口并重复补发。
    - 同一套语义同时服务非共卡全局 progress 和共卡 produce_batch 的局部 progress。

    使用注意：
    - 不要在 strategy 中补 key 或用 dict.get(..., 0) 兜底；缺少 task key 应 fail fast。
    - 除非语义明确要求冻结本轮 produce_batch 的 target / scheduled_target，
      否则不要把字段值复制成局部快照后跨 await 使用；需要字段值时直接读 progress.xxx，
      让并发更新后的 next_consumer_step / consumed_samples 能尽早生效。
    - 运行中不要整体替换 ProduceProgress 对象；resume 时也应原地更新字段，避免旧引用失效。

    字段含义：
    - next_consumer_step：producer 写入新样本时应面向的训练 step。get_batch(i) 入口设为 i，
      成功取出非空 batch 后设为 i + 1。
    - producer_future_step：producer 当前准备生产的 future step。
    - consumed_samples：各 task 已被 consumer 从 replay buffer 取走的累计计数。
      单位由当前 task 的 strategy ``count_unit`` 决定：legacy shim 下是 group 数，
      progressive 下是 trajectory 数。
    - target_samples：各 task 截至 target_upto_future_step 应生产出的累计目标，单位同上。
    - target_upto_future_step：target_samples 已覆盖到的最大 future step。
    - stopped_prompts：progressive 模式下各 task 因 max_repeat 耗尽被丢弃的 prompt 累计数。
    - needs_more_reentries：progressive 模式下各 task 触发 NEEDS_MORE 队首重入的累计次数。
    """

    next_consumer_step: int = 1
    producer_future_step: int = 1
    consumed_samples: dict[str, int] = field(default_factory=dict)
    target_samples: dict[str, int] = field(default_factory=dict)
    target_upto_future_step: int = 0
    stopped_prompts: dict[str, int] = field(default_factory=dict)
    needs_more_reentries: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, task_names: list[str]) -> "ProduceProgress":
        return cls(
            consumed_samples={task_name: 0 for task_name in task_names},
            target_samples={task_name: 0 for task_name in task_names},
            stopped_prompts={task_name: 0 for task_name in task_names},
            needs_more_reentries={task_name: 0 for task_name in task_names},
        )

    @classmethod
    def build_local(
        cls,
        task_names: list[str],
        task_batch_sizes: dict[str, int],
        train_step: int,
    ) -> "ProduceProgress":
        # 共卡路径使用局部 progress，只表达本次 produce_batch 的目标，不污染非共卡累计窗口。
        return cls(
            next_consumer_step=train_step,
            producer_future_step=train_step,
            consumed_samples={task_name: 0 for task_name in task_names},
            target_samples=dict(task_batch_sizes),
            target_upto_future_step=train_step,
            stopped_prompts={task_name: 0 for task_name in task_names},
            needs_more_reentries={task_name: 0 for task_name in task_names},
        )

    def ensure_target_upto(
        self,
        *,
        batch_size: int,
        future_step: int,
        allocate_batch_sizes: Callable[[int, int], dict[str, int]],
    ) -> dict[str, int]:
        """把累计 target 推进到指定 future step，并返回该 step 的 task batch size。"""

        if future_step > self.target_upto_future_step:
            for step in range(self.target_upto_future_step + 1, future_step + 1):
                task_batch_sizes = allocate_batch_sizes(batch_size, step)
                for task_name, task_batch_size in task_batch_sizes.items():
                    self.target_samples[task_name] += task_batch_size
            self.target_upto_future_step = future_step

        return allocate_batch_sizes(batch_size, future_step)

    def begin_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step

    def mark_consumed(self, consumed_counts: dict[str, int]) -> None:
        # consumer 真实取出多少就累计多少，target 不回退，避免 producer 把已消费样本当成缺口。
        for task_name, count in consumed_counts.items():
            self.consumed_samples[task_name] += count

    def mark_stopped(self, task_name: str, count: int = 1) -> None:
        self.stopped_prompts[task_name] = self.stopped_prompts.get(task_name, 0) + count

    def mark_needs_more(self, task_name: str, count: int = 1) -> None:
        self.needs_more_reentries[task_name] = self.needs_more_reentries.get(task_name, 0) + count

    def finish_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step + 1

    def advance_future_step(self) -> None:
        self.producer_future_step += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_consumer_step": self.next_consumer_step,
            "producer_future_step": self.producer_future_step,
            "consumed_samples": dict(self.consumed_samples),
            "target_samples": dict(self.target_samples),
            "target_upto_future_step": self.target_upto_future_step,
            "stopped_prompts": dict(self.stopped_prompts),
            "needs_more_reentries": dict(self.needs_more_reentries),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # 原地更新 dict，避免 strategy / context 持有旧引用。
        self.next_consumer_step = state["next_consumer_step"]
        self.producer_future_step = state["producer_future_step"]
        self.target_upto_future_step = state["target_upto_future_step"]
        self.consumed_samples.clear()
        self.consumed_samples.update(state["consumed_samples"])
        self.target_samples.clear()
        self.target_samples.update(state["target_samples"])
        self.stopped_prompts.clear()
        self.stopped_prompts.update(state.get("stopped_prompts", {}))
        self.needs_more_reentries.clear()
        self.needs_more_reentries.update(state.get("needs_more_reentries", {}))


class ProduceBatchStatus(Enum):
    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return True


def default_should_continue_fn(completed_count: int, batch_size: int, **kwargs) -> bool:
    return completed_count < batch_size


def calculate_stale_threshold(max_staleness: int, sync_weights_interval: int) -> int:
    if max_staleness < 0:
        raise ValueError(f"max_staleness must be non-negative, got {max_staleness}.")
    if sync_weights_interval <= 0:
        raise ValueError(f"sync_weights_interval must be positive, got {sync_weights_interval}.")

    # max_staleness 按同步周期计数；+1 表示训练天然必须接受的当前同步周期滞后。
    return (max_staleness + 1) * sync_weights_interval


@runtime_checkable
class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[RolloutState]) -> bool: ...


@runtime_checkable
class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, batch_size: int, **kwargs) -> bool: ...


CountUnit = Literal["groups", "trajectories"]


@dataclass
class ProduceContext:
    """单 task 生产上下文。

    这里集中维护 AsyncProduceStrategy 最容易传错的运行时契约：
    - strategy 只接受 ProduceContext，不再兼容散装参数入口；
    - target / consumed 都按绝对累计口径读取；
    - 暂停只读 manager 传入的 update_event；
    - rollout generate 的 ray/local 差异和 timing 字段写入；
    - 生成结果先按业务有效性过滤，再统一交给 ReplayBuffer 写版本、刷新 staleness、执行过期。
    """

    agent_loop: AgentLoopSpec
    sampler: Sampler
    replay_buffer: ReplayBuffer
    task_batch_size: int
    task_name: str
    train_step: int
    update_event: asyncio.Event
    model_step: int
    progress: ProduceProgress
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    stale_threshold: int | None = None

    @property
    def consumer_step(self) -> int:
        return self.progress.next_consumer_step

    @property
    def target_abs(self) -> int:
        return self.progress.target_samples[self.task_name]

    def should_abort(self) -> bool:
        return self.update_event.is_set()

    async def expired_count(self) -> int:
        return await self.replay_buffer.count(task_name=self.task_name, group_status=Status.EXPIRED)

    async def available_count(self) -> int:
        completed_count = await self.replay_buffer.count(task_name=self.task_name, group_status=Status.COMPLETED)
        return self.progress.consumed_samples[self.task_name] + completed_count

    async def available_trajectory_count(self) -> int:
        """Trajectory-unit analogue of :meth:`available_count`.

        Sums the in-buffer COMPLETED trajectories plus the cumulative
        ``consumed_samples`` counter for this task. Callers using
        ``count_unit="trajectories"`` must populate ``consumed_samples``
        with trajectory counts for this arithmetic to line up.
        """
        completed_trajectories = await self.replay_buffer.count_trajectories(
            task_name=self.task_name, group_status=Status.COMPLETED
        )
        return self.progress.consumed_samples[self.task_name] + completed_trajectories

    async def sample_group(self, *, from_expired_pool: bool) -> list[RolloutState]:
        group_status = [Status.EXPIRED, Status.ABORTED] if from_expired_pool else [Status.ABORTED]
        return await self.sampler.sample(task_name=self.task_name, group_status=group_status)

    async def generate_group(
        self,
        rollout_state: list[RolloutState],
        *,
        enable_partial_rollout: bool = False,
    ) -> list[RolloutState]:
        # strategy 只表达“要生成”，不关心 agent_loop 是 ray actor 还是本地对象。
        start = time.perf_counter()
        if isinstance(self.agent_loop, ray.actor.ActorHandle):
            result = await self.agent_loop.generate_group.remote(
                rollout_state,
                enable_partial_rollout=enable_partial_rollout,
            )
        else:
            result = await self.agent_loop.generate_group(
                rollout_state,
                enable_partial_rollout=enable_partial_rollout,
            )
        elapsed = time.perf_counter() - start
        for item in result:
            extra_fields = getattr(item, "extra_fields", None)
            if extra_fields is None:
                extra_fields = {}
                setattr(item, "extra_fields", extra_fields)
            extra_fields[GROUP_GENERATE_TIME_KEY] = elapsed
        return result

    async def put_generated_group(self, group: list[RolloutState]) -> bool:
        # 只有完整生成的 group 才需要业务有效性过滤；ABORTED / EXPIRED 保留原状态供重试或统计。
        initial_status = get_group_status(group)
        group_size = len(group)
        sample_uids: list[int | None] = [getattr(item, "uid", None) for item in group[:3]]
        prompt_uid = getattr(group[0], "message_uid", None) if group else None
        logger.debug(
            f"[{self.task_name}] put_generated_group: initial_status={initial_status.name}, "
            f"size={group_size}, prompt_uid={prompt_uid}, sample_uids={sample_uids}"
        )
        is_completed = initial_status == Status.COMPLETED
        if is_completed:
            is_valid = self.is_valid_sample_fn(group)
            if not is_valid:
                logger.debug(
                    f"[{self.task_name}] group filtered by is_valid_sample_fn: "
                    f"prompt_uid={prompt_uid}, size={group_size}"
                )
                for item in group:
                    item.status = Status.FILTERED
        await self.replay_buffer.put(
            group,
            self.task_name,
            model_step=self.model_step,
            current_train_step=self.consumer_step,
            stale_threshold=self.stale_threshold,
        )
        # replay_buffer.put 可能把 stale group 转为 EXPIRED，返回前重新判断是否仍可训练。
        final_status = get_group_status(group)
        if final_status != initial_status:
            logger.debug(
                f"[{self.task_name}] group status transition: "
                f"{initial_status.name} -> {final_status.name} "
                f"(prompt_uid={prompt_uid}, size={group_size})"
            )
        return final_status == Status.COMPLETED


class ProduceStrategyConfig(ABC, BaseModel):
    """Base configuration for rollout production strategies.

    Production strategies decide how the agent loop fills the replay buffer and
    when it should stop producing samples for the current training step.

    Args:
        is_valid_sample_fn (IsValidSampleFn): Function used to decide whether a
            generated rollout group is trainable. Defaults to
            ``default_is_valid_sample_fn``.
        should_continue_fn (ShouldContinueFn): Function used to decide whether
            production should continue after a group is processed. Defaults to
            ``default_should_continue_fn``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        prompt_repeat_k: int = 1,
        judgers: dict[str, Judger] | None = None,
    ) -> "ProduceStrategy": ...


class SyncProduceStrategyConfig(ProduceStrategyConfig):
    """Legacy shim: colocated / synchronous path.

    Translates to a :class:`TrajectoryProduceStrategy` with
    ``wait_until_all_ready=True`` and ``max_staleness=0``. The aggregator
    runs with ``min_repeat == max_repeat == prompt_repeat_k`` and
    ``stop_when_all_equal=False`` so READY fires on the k-th completion
    regardless of reward variance, matching legacy group behavior.
    """

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        prompt_repeat_k: int = 1,
        judgers: dict[str, Judger] | None = None,
    ) -> "ProduceStrategy":
        return _build_trajectory_strategy(
            wait_until_all_ready=True,
            over_sample_threshold=0.0,
            enable_partial_rollout=False,
            max_staleness=0,
            tail_batch_trigger_size=0,
            sync_weights_interval=sync_weights_interval,
            prompt_repeat_k=prompt_repeat_k,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
            judgers=judgers,
        )


class AsyncProduceStrategyConfig(ProduceStrategyConfig):
    """Legacy shim: disaggregated / async path.

    Preserves the legacy knobs (``over_sample_threshold``,
    ``enable_partial_rollout``, ``max_staleness``, ``tail_batch_trigger_size``)
    and routes them onto :class:`TrajectorySchedulerConfig`. The group
    policy is fixed to min=max=prompt_repeat_k with
    ``stop_when_all_equal=False`` for behavioral equivalence with the
    removed AsyncProduceStrategy class body.
    """

    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = Field(default=0, ge=0)
    tail_batch_trigger_size: int = 0

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        prompt_repeat_k: int = 1,
        judgers: dict[str, Judger] | None = None,
    ) -> "ProduceStrategy":
        return _build_trajectory_strategy(
            wait_until_all_ready=False,
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            max_staleness=self.max_staleness,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            sync_weights_interval=sync_weights_interval,
            prompt_repeat_k=prompt_repeat_k,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
            judgers=judgers,
        )


def _build_trajectory_strategy(
    *,
    wait_until_all_ready: bool,
    over_sample_threshold: float,
    enable_partial_rollout: bool,
    max_staleness: int,
    tail_batch_trigger_size: int,
    sync_weights_interval: int,
    prompt_repeat_k: int,
    is_valid_sample_fn: IsValidSampleFn,
    should_continue_fn: ShouldContinueFn,
    count_unit: CountUnit = "groups",
    judgers: dict[str, Judger] | None = None,
) -> "TrajectoryProduceStrategy":
    group_policy = GroupPolicyConfig(
        min_repeat=prompt_repeat_k,
        max_repeat=prompt_repeat_k,
        stop_when_all_equal=False,
    ).build()
    scheduler_cfg = TrajectorySchedulerConfig(
        max_on_fly=max(1, prompt_repeat_k * 64),
        wait_until_all_ready=wait_until_all_ready,
        max_staleness=max_staleness,
        enable_partial_rollout=enable_partial_rollout,
        tail_batch_trigger_size=tail_batch_trigger_size,
    )
    scheduler = scheduler_cfg.build(sync_weights_interval=sync_weights_interval)
    return TrajectoryProduceStrategy(
        scheduler=scheduler,
        group_policy=group_policy,
        over_sample_threshold=over_sample_threshold,
        is_valid_sample_fn=is_valid_sample_fn,
        should_continue_fn=should_continue_fn,
        count_unit=count_unit,
        judgers=judgers,
    )


class ProgressiveProduceStrategyConfig(ProduceStrategyConfig):
    """Progressive-sampling strategy with explicit group-policy / scheduler knobs.

    Unlike :class:`SyncProduceStrategyConfig` and
    :class:`AsyncProduceStrategyConfig`, this config exposes
    :class:`GroupPolicyConfig` directly so users can configure
    ``min_repeat < max_repeat`` and enable the all-equal NEEDS_MORE /
    STOPPED state machine.

    ``ProduceProgress.target_samples`` and
    ``ProduceProgress.consumed_samples`` are interpreted as trajectory
    counts when this config is used; the manager switches to
    :meth:`ReplayBuffer.take_batch_by_trajectory_count` and allows the
    last group to overflow the target.

    Args:
        group_policy (GroupPolicyConfig): Policy controlling
            COLLECTING / NEEDS_MORE / READY / STOPPED transitions.
        scheduler (TrajectorySchedulerConfig): Scheduler capacity and
            partial-rollout behavior.
        over_sample_threshold (float): Extra prompts kept preloaded beyond
            the strict target (same semantic as legacy but in the same unit
            as the config's ``count_unit``, which is always
            ``"trajectories"`` here).
    """

    group_policy: GroupPolicyConfig
    scheduler: TrajectorySchedulerConfig
    over_sample_threshold: float = 0.0

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        prompt_repeat_k: int = 1,
        judgers: dict[str, Judger] | None = None,
    ) -> "ProduceStrategy":
        # prompt_repeat_k coming from SamplerConfig is ignored: the
        # progressive config expresses its own min/max via group_policy.
        _ = prompt_repeat_k
        scheduler = self.scheduler.build(sync_weights_interval=sync_weights_interval)
        return TrajectoryProduceStrategy(
            scheduler=scheduler,
            group_policy=self.group_policy.build(),
            over_sample_threshold=self.over_sample_threshold,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
            count_unit="trajectories",
            judgers=judgers,
        )


class ProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    @abstractmethod
    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: ProduceContext) -> float:
        return 0.0

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return False

    def pending_task_count(self) -> int:
        return 0

    async def state_dict(self) -> dict[str, Any]:
        return {}

    async def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


class TrajectoryProduceStrategy(ProduceStrategy):
    """Trajectory-level producer backed by TrajectoryScheduler + GroupAggregator.

    The strategy preloads prompts onto the scheduler queue, spawns trajectory
    runners up to the scheduler's global cap, and forwards completed groups
    to the replay buffer via :meth:`ProduceContext.put_generated_group`.

    Each runner:

    1. Deep-copies the prompt template and assigns a fresh trajectory uid.
    2. Invokes ``ctx.generate_group`` on a single-element list so
       per-trajectory and batch-judge agent loops both keep working.
    3. Updates the aggregator atomically: on READY the completed group is
       written to the buffer; on COLLECTING / NEEDS_MORE the prompt is
       re-submitted (NEEDS_MORE goes to the front); on STOPPED the
       aggregation is dropped and the per-task telemetry counter ticks.

    ``count_unit`` chooses the semantic of ``ProduceProgress.target_samples``
    and ``ProduceProgress.consumed_samples`` for this strategy:

    * ``"groups"`` (legacy Sync / Async shims): target and consumed counters
      track group counts. The produce loop exits when
      ``ctx.available_count()`` reaches the target.
    * ``"trajectories"`` (progressive config): counters track trajectory
      counts and the loop uses ``ctx.available_trajectory_count()`` so that
      variable-K groups fall out naturally.

    Args:
        scheduler (TrajectoryScheduler): Global concurrency / queue manager.
        group_policy (GroupPolicy): Policy driving aggregation decisions.
        over_sample_threshold (float): Extra prompts kept preloaded beyond
            the strict target, mirroring the legacy knob. Interpreted in
            the same unit as ``count_unit``.
        is_valid_sample_fn (IsValidSampleFn): Filter applied on finalized
            groups before they are written to the replay buffer.
        should_continue_fn (ShouldContinueFn): Loop termination predicate
            evaluated against ``available`` vs. ``target``.
        count_unit (CountUnit): ``"groups"`` or ``"trajectories"``. See
            above.
    """

    def __init__(
        self,
        *,
        scheduler: TrajectoryScheduler,
        group_policy: GroupPolicy,
        over_sample_threshold: float,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
        count_unit: CountUnit = "groups",
        judgers: dict[str, Judger] | None = None,
    ) -> None:
        super().__init__(is_valid_sample_fn, should_continue_fn)
        self._scheduler = scheduler
        self._policy = group_policy
        self._aggregator = GroupAggregator(group_policy)
        self._over_sample_threshold = over_sample_threshold
        self._count_unit: CountUnit = count_unit
        self._judgers: dict[str, Judger] = judgers or {}

    @property
    def stale_threshold(self) -> int:
        return self._scheduler.stale_threshold

    @property
    def aggregator(self) -> GroupAggregator:
        return self._aggregator

    @property
    def scheduler(self) -> TrajectoryScheduler:
        return self._scheduler

    @property
    def count_unit(self) -> CountUnit:
        return self._count_unit

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return self._scheduler.is_model_expired(train_step, model_step)

    def pending_task_count(self) -> int:
        return self._scheduler.pending_count()

    def _resolve_batch_judger(self, state: RolloutState) -> Judger | None:
        if not self._judgers:
            return None
        data_source = state.data_source
        if data_source is None:
            return None
        judger = self._judgers.get(data_source)
        if judger is None or not judger.is_batch_judger:
            return None
        return judger

    async def pause_produce(self, ctx: ProduceContext) -> float:
        pause_start = time.perf_counter()
        if self._scheduler.pending_count() == 0:
            # Drain stale queued PromptRequests so the next produce_batch
            # does not see double-bookkeeping (queue + aggregator). The
            # aggregator state is preserved across steps so unfinished
            # aggregations can be resumed in _resubmit_unfinished_aggregations.
            queue_drained = len(await self._scheduler.clear_queue())
            if queue_drained:
                logger.info(
                    f"[{ctx.task_name}] pause_produce idle cleanup: "
                    f"queue_drained={queue_drained}"
                )
            return 0.0
        rollout_ctl = await get_agent_loop_rollout_ctl(ctx.agent_loop)
        await pause_generation(rollout_ctl)
        await self._scheduler.pause_and_cleanup()
        # Clear queued prompts that never got spawned this round; the
        # aggregations they registered are still live in the aggregator
        # and will be re-submitted by _resubmit_unfinished_aggregations
        # next produce_batch via a single canonical path. The aggregator
        # itself is intentionally NOT cleared — completed / pending_keep
        # trajectories represent real inference work that should be
        # reused across steps.
        queue_drained = len(await self._scheduler.clear_queue())
        unfinished = await self._aggregator.active_count()
        if queue_drained or unfinished:
            logger.info(
                f"[{ctx.task_name}] pause_produce cleanup: "
                f"queue_drained={queue_drained}, "
                f"unfinished_aggregations_kept={unfinished}"
            )
        return time.perf_counter() - pause_start

    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus:
        if ctx.task_name not in ctx.progress.consumed_samples:
            raise KeyError(f"ProduceProgress.consumed_samples missing task_name={ctx.task_name!r}")
        if ctx.task_name not in ctx.progress.target_samples:
            raise KeyError(f"ProduceProgress.target_samples missing task_name={ctx.task_name!r}")

        if ctx.target_abs <= 0:
            return ProduceBatchStatus.NORMAL

        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.train_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        stopped_start = ctx.progress.stopped_prompts.get(ctx.task_name, 0)
        needs_more_start = ctx.progress.needs_more_reentries.get(ctx.task_name, 0)

        logger.info(
            f"[{ctx.task_name}] produce_batch start: "
            f"target={ctx.target_abs} ({self._count_unit}), "
            f"task_batch_size={ctx.task_batch_size}, "
            f"over_sample_threshold={self._over_sample_threshold}, "
            f"max_on_fly={self._scheduler.config.max_on_fly}, "
            f"stop_when_all_equal={self._policy.config.stop_when_all_equal}, "
            f"policy=[{self._policy.min_repeat}-{self._policy.max_repeat}], "
            f"stopped_so_far={stopped_start}, needs_more_so_far={needs_more_start}"
        )

        # Cross-step reuse: aggregations whose in-flight trajectories were
        # cancelled by the previous step's pause_produce sit idle in the
        # aggregator. Re-evaluate each (refresh staleness, finalise if
        # already READY, otherwise re-submit prompts to fill the gap to
        # min_repeat) before pulling fresh prompts from the dataloader.
        resume_stats = await self._resubmit_unfinished_aggregations(ctx)
        if any(resume_stats.values()):
            logger.info(
                f"[{ctx.task_name}] cross-step resume: "
                f"resubmitted={resume_stats['resubmitted']} "
                f"(partial_resume={resume_stats['resubmitted_partial_resume']}, "
                f"collecting={resume_stats['resubmitted_collecting']}, "
                f"needs_more={resume_stats['resubmitted_needs_more']}), "
                f"finalized={resume_stats['finalized']}, "
                f"stale_completed={resume_stats['stale_completed']}, "
                f"stale_pending={resume_stats['stale_pending']}, "
                f"queue_after_resume=[{format_queue_breakdown(self._scheduler.queue_lens_by_priority())}]"
            )

        runner = self._build_pipeline(ctx)
        iteration = 0
        last_log_time = time.perf_counter()
        exit_reason = "normal"

        while True:
            iteration += 1
            if ctx.should_abort():
                exit_reason = "abort"
                status_out = ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
                break
            if self.is_model_expired(ctx.train_step, ctx.model_step):
                exit_reason = "model_expired"
                status_out = ProduceBatchStatus.EXPIRED_BATCH
                break

            available = await self._available_for_unit(ctx)
            if not self.should_continue_fn(available, ctx.target_abs):
                exit_reason = f"target_met (available={available}, target={ctx.target_abs})"
                status_out = ProduceBatchStatus.NORMAL
                break

            await self._preload_prompts(ctx, available)

            spawned_any = False
            while await self._scheduler.spawn_if_slot(runner):
                spawned_any = True

            if not spawned_any and self._scheduler.pending_count() == 0:
                if (
                    self._scheduler.queue_len() == 0
                    and await self._aggregator.active_count() == 0
                ):
                    logger.warning(
                        f"[{ctx.task_name}] produce stalled: available={available}/"
                        f"{ctx.target_abs}, no pending tasks, empty queue, no active aggregation. "
                        f"stopped_this_call={ctx.progress.stopped_prompts.get(ctx.task_name, 0) - stopped_start}, "
                        f"needs_more_this_call={ctx.progress.needs_more_reentries.get(ctx.task_name, 0) - needs_more_start}."
                    )
                    exit_reason = f"stalled (available={available}, target={ctx.target_abs})"
                    status_out = ProduceBatchStatus.NORMAL
                    break

            now = time.perf_counter()
            if now - last_log_time >= PRODUCE_PROGRESS_LOG_INTERVAL_S:
                logger.info(
                    f"[{ctx.task_name}] iter={iteration} available={available}/{ctx.target_abs} "
                    f"inflight={self._scheduler.inflight_count()} "
                    f"pending={self._scheduler.pending_count()} "
                    f"queue={self._scheduler.queue_len()} "
                    f"queue_by_pri=[{format_queue_breakdown(self._scheduler.queue_lens_by_priority())}] "
                    f"active_aggregations={await self._aggregator.active_count()} "
                    f"stopped_this_call={ctx.progress.stopped_prompts.get(ctx.task_name, 0) - stopped_start} "
                    f"needs_more_this_call={ctx.progress.needs_more_reentries.get(ctx.task_name, 0) - needs_more_start}"
                )
                last_log_time = now

            await self._scheduler.wait_first_completed(timeout_s=1.0)

        if self._scheduler.config.wait_until_all_ready:
            await self._scheduler.drain()

        stopped_delta = ctx.progress.stopped_prompts.get(ctx.task_name, 0) - stopped_start
        needs_more_delta = ctx.progress.needs_more_reentries.get(ctx.task_name, 0) - needs_more_start
        logger.info(
            f"[{ctx.task_name}] produce_batch exit: reason={exit_reason}, "
            f"iterations={iteration}, stopped={stopped_delta}, needs_more={needs_more_delta}, "
            f"pending_remaining={self._scheduler.pending_count()}, "
            f"queue_remaining={self._scheduler.queue_len()}, "
            f"queue_by_pri=[{format_queue_breakdown(self._scheduler.queue_lens_by_priority())}], "
            f"active_aggregations={await self._aggregator.active_count()}"
        )
        return status_out

    async def state_dict(self) -> dict[str, Any]:
        return {
            "aggregator": await self._aggregator.state_dict(),
        }

    async def load_state_dict(self, state: dict[str, Any]) -> None:
        if "aggregator" in state:
            await self._aggregator.load_state_dict(state["aggregator"])

    async def _available_for_unit(self, ctx: ProduceContext) -> int:
        # Main loop's exit condition only counts trajectories that have
        # actually landed in the replay buffer, because that's what
        # ``take_batch`` will pull from. Trajectories sitting in
        # aggregator.completed have not yet been finalised into a group
        # — they need their sibling in-flight trajectories to finish so
        # add_trajectory can flip the aggregation to READY → buffer.
        # The Q3 "don't over-sample fresh prompts when partial progress
        # already covers target" semantics is enforced inside
        # ``_preload_saturating`` via a target-gap cap, not here.
        if self._count_unit == "trajectories":
            return await ctx.available_trajectory_count()
        return await ctx.available_count()

    async def _resubmit_unfinished_aggregations(
        self, ctx: ProduceContext
    ) -> dict[str, int]:
        """Resume aggregations whose in-flight trajectories were cancelled.

        Runs once at the top of each colocated produce_batch. For every
        aggregation still alive in the aggregator (carried over from the
        previous step's pause_produce), this method:

        1. Refreshes per-trajectory ``seq_staleness`` against the current
           ``train_step`` and drops stale completed / pending_keep entries
           (``routed_experts`` ObjectRefs are freed inside the aggregator).
        2. Re-runs the policy: if the surviving completed list already
           satisfies READY, finalise the group straight to the replay
           buffer (no extra inference needed).
        3. Otherwise re-submits PromptRequests to ``submit_front`` so the
           scheduler picks them up before any fresh prompts pulled by
           ``_preload_saturating``. The resubmit count covers:
           - one slot per ``pending_keep`` entry — runners with
             ``enable_partial_rollout=True`` will pop them and resume the
             partial trajectory.
           - additional fresh spawns to fill the gap to ``min_repeat``
             (COLLECTING) or one ``min_repeat`` chunk for the NEEDS_MORE
             retry path. Total commitment is capped by ``max_repeat``.

        The producer's main loop sees the resumed aggregations through
        ``aggregator.completed_trajectory_count`` (Q3) so ``available_for_unit``
        already reflects the partial progress.
        """
        snapshots = await self._aggregator.list_unfinished_snapshots()
        stats = {
            "resubmitted": 0,
            "resubmitted_partial_resume": 0,
            "resubmitted_collecting": 0,
            "resubmitted_needs_more": 0,
            "finalized": 0,
            "stale_completed": 0,
            "stale_pending": 0,
        }
        if not snapshots:
            return stats

        for snap in snapshots:
            stale_completed, stale_pending = await self._aggregator.refresh_aggregation_staleness(
                snap.prompt_uid, ctx.train_step, self.stale_threshold
            )
            stats["stale_completed"] += stale_completed
            stats["stale_pending"] += stale_pending

            finalized_group = await self._aggregator.try_finalize_if_ready(snap.prompt_uid)
            if finalized_group is not None:
                batch_judger = self._resolve_batch_judger(finalized_group[0])
                if batch_judger is not None:
                    finalized_group = await batch_judger.judge(finalized_group)
                await ctx.put_generated_group(finalized_group)
                stats["finalized"] += 1
                continue

            fresh = await self._aggregator.get_snapshot(snap.prompt_uid)
            if fresh is None:
                continue

            n_completed = len(fresh.completed)
            n_pending = len(fresh.pending_keep)
            committed = n_completed + n_pending
            gap_max = max(0, self._policy.max_repeat - committed)

            if n_completed < self._policy.min_repeat:
                # COLLECTING: spawn enough new trajectories so total
                # commitment (completed + pending + new) reaches min_repeat.
                n_new_spawn = max(0, self._policy.min_repeat - committed)
                # NEEDS_MORE distinction below applies only when completed
                # >= min_repeat; here the new spawns are still COLLECTING.
                new_spawn_priority = PRIORITY_COLLECTING
            else:
                # completed >= min_repeat but not READY (try_finalize_if_ready
                # already caught READY above). Treat as NEEDS_MORE: spawn
                # one min_repeat-sized chunk capped by remaining headroom.
                n_new_spawn = min(self._policy.min_repeat, gap_max)
                new_spawn_priority = PRIORITY_NEEDS_MORE

            # Resume slots first (pending_keep), then fresh siblings. Both
            # go through submit_front but into different priority buckets,
            # so the scheduler will drain pending_keep first within this
            # prompt and across all prompts.
            for _ in range(n_pending):
                await self._scheduler.submit_front(
                    PromptRequest(
                        prompt_uid=fresh.prompt_uid,
                        task_name=ctx.task_name,
                        prompt=fresh.original_prompt,
                        priority=PRIORITY_PARTIAL_RESUME,
                    )
                )
            for _ in range(n_new_spawn):
                await self._scheduler.submit_front(
                    PromptRequest(
                        prompt_uid=fresh.prompt_uid,
                        task_name=ctx.task_name,
                        prompt=fresh.original_prompt,
                        priority=new_spawn_priority,
                    )
                )
            stats["resubmitted"] += n_pending + n_new_spawn
            stats["resubmitted_partial_resume"] += n_pending
            if new_spawn_priority == PRIORITY_NEEDS_MORE:
                stats["resubmitted_needs_more"] += n_new_spawn
            else:
                stats["resubmitted_collecting"] += n_new_spawn

        return stats

    async def _preload_prompts(self, ctx: ProduceContext, available: int) -> None:
        """Top up the scheduler queue with fresh prompts.

        Trajectory mode (progressive) is *saturation-driven*: the producer
        keeps pulling prompts from the dataloader until
        ``pending + queue`` equals the scheduler's ``max_on_fly`` cap, so
        the inference engine stays busy regardless of how close
        ``available`` is to ``target_abs``. Any group finalised past the
        target simply lands in the replay buffer for the next training
        step; ``max_staleness`` and partial rollout absorb the cross-step
        reuse. The legacy deficit-based behaviour is kept for ``groups``
        mode so the Sync / Async shim stays byte-equivalent.
        """
        if self._count_unit == "trajectories":
            await self._preload_saturating(ctx)
            return
        await self._preload_deficit_based(ctx, available)

    async def _preload_saturating(self, ctx: ProduceContext) -> None:
        per_prompt = max(1, self._policy.min_repeat)
        pending = self._scheduler.pending_count()
        queue = self._scheduler.queue_len()
        capacity = self._scheduler.config.max_on_fly
        current_load = pending + queue
        slots_available = max(0, capacity - current_load)
        if slots_available < per_prompt:
            logger.debug(
                f"[{ctx.task_name}] preload skipped: saturated "
                f"(pending={pending}, queue={queue}, capacity={capacity})"
            )
            return
        prompts_to_add = slots_available // per_prompt
        # No additional target-gap cap: ``slots_available = max_on_fly -
        # (pending + queue)`` already guarantees ``inflight + queue <=
        # max_on_fly``, so the inference engine stays saturated. A cap
        # based on ``target_with_oversample`` would actively contradict
        # this when the user sets ``max_on_fly > target_with_oversample``
        # — the producer would refuse to keep the rollout engine fed
        # even though inflight is below the configured cap. Cross-step
        # reuse already shows up in ``queue`` (cross-step resume puts
        # PromptRequests there), so saturating's existing slots_available
        # cap implicitly avoids over-sampling fresh prompts on top of
        # carry-over commitments.
        if prompts_to_add <= 0:
            return
        logger.debug(
            f"[{ctx.task_name}] preload (saturating): pending={pending}, queue={queue}, "
            f"queue_by_pri=[{format_queue_breakdown(self._scheduler.queue_lens_by_priority())}], "
            f"capacity={capacity}, slots_available={slots_available}, "
            f"per_prompt={per_prompt}, prompts_to_add={prompts_to_add}"
        )
        preloaded = 0
        for _ in range(prompts_to_add):
            if ctx.should_abort():
                logger.debug(
                    f"[{ctx.task_name}] preload aborted after {preloaded}/{prompts_to_add} prompts"
                )
                return
            try:
                prompt_req = await ctx.sampler.sample_prompt(task_name=ctx.task_name)
            except SamplerExhausted:
                # Single-epoch sampler (e.g. eval) has yielded its last item.
                # Stop topping up the queue; the produce_batch main loop's
                # stalled-detection path will exit once the in-flight tasks
                # drain. This guarantees the producer never sends more
                # trajectories than the dataset contains.
                logger.debug(
                    f"[{ctx.task_name}] sampler exhausted after {preloaded} preloads "
                    f"this round; stopping further preload."
                )
                return
            await self._aggregator.register_prompt(
                prompt_req.prompt,
                ctx.task_name,
                is_batch_judger=self._resolve_batch_judger(prompt_req.prompt) is not None,
            )
            for _ in range(per_prompt):
                await self._scheduler.submit(prompt_req)
            preloaded += 1

    async def _preload_deficit_based(self, ctx: ProduceContext, available: int) -> None:
        deficit = ctx.target_abs - available
        if deficit <= 0:
            return
        per_prompt = 1
        prompts_for_deficit = deficit
        oversample_prompts = math.ceil(self._over_sample_threshold * ctx.task_batch_size)
        active_prompts = await self._aggregator.active_count()
        target_active = prompts_for_deficit + oversample_prompts
        to_preload = max(0, target_active - active_prompts)
        if to_preload <= 0:
            logger.debug(
                f"[{ctx.task_name}] preload skipped: deficit={deficit} (groups), "
                f"prompts_for_deficit={prompts_for_deficit}, oversample_prompts={oversample_prompts}, "
                f"active_prompts={active_prompts}"
            )
            return
        logger.debug(
            f"[{ctx.task_name}] preload: deficit={deficit} (groups), "
            f"per_prompt={per_prompt}, prompts_for_deficit={prompts_for_deficit}, "
            f"oversample_prompts={oversample_prompts}, active_prompts={active_prompts}, "
            f"to_preload={to_preload}, initial_submits_per_prompt=1"
        )
        preloaded = 0
        for _ in range(to_preload):
            if ctx.should_abort():
                logger.debug(
                    f"[{ctx.task_name}] preload aborted after {preloaded}/{to_preload} prompts"
                )
                return
            try:
                prompt_req = await ctx.sampler.sample_prompt(task_name=ctx.task_name)
            except SamplerExhausted:
                logger.info(
                    f"[{ctx.task_name}] sampler exhausted after {preloaded} preloads "
                    f"this round; stopping further preload."
                )
                return
            await self._aggregator.register_prompt(
                prompt_req.prompt,
                ctx.task_name,
                is_batch_judger=self._resolve_batch_judger(prompt_req.prompt) is not None,
            )
            await self._scheduler.submit(prompt_req)
            preloaded += 1

    def _build_pipeline(self, ctx: ProduceContext) -> Pipeline:
        """Build the per-prompt pipeline passed to the scheduler.

        The pipeline has two phases. The inference phase (deepcopy +
        ``generate_group``) runs while the scheduler's ``max_on_fly`` slot
        is held; the post phase (aggregator + replay buffer dispatch) runs
        after :func:`release_slot` so a new inference can be spawned in
        parallel. ``mark_in_flight`` brackets the whole pipeline because
        the policy's ``should_spawn_more`` reads the in-flight count to
        cap commitment at ``max_repeat``.
        """

        async def pipeline(req: PromptRequest, release_slot: Callable[[], None]) -> None:
            await self._aggregator.mark_in_flight(req.prompt_uid, +1)
            try:
                # Phase 1: inference (holds a max_on_fly slot)
                pending: RolloutState | None = None
                if self._scheduler.config.enable_partial_rollout:
                    pending = await self._aggregator.pop_pending_keep(req.prompt_uid)
                if pending is not None:
                    input_state = pending
                    logger.debug(
                        f"[{ctx.task_name}] runner resuming prompt_uid={req.prompt_uid} "
                        f"from pending_keep traj_uid={pending.uid}"
                    )
                else:
                    input_state = copy.deepcopy(req.prompt)
                    input_state.uid = uuid4().int
                    input_state.session_uid = input_state.uid
                    logger.debug(
                        f"[{ctx.task_name}] runner spawning prompt_uid={req.prompt_uid} "
                        f"-> traj_uid={input_state.uid}"
                    )
                try:
                    result_list = await ctx.generate_group(
                        [input_state],
                        enable_partial_rollout=self._scheduler.config.enable_partial_rollout,
                    )
                finally:
                    # Slot is freed regardless of success / failure so the
                    # next inference can start while we run the post phase.
                    release_slot()
                traj = result_list[0]
                # Phase 2: dispatch (no slot held)
                await self._dispatch_trajectory(ctx, req, traj)
            finally:
                await self._aggregator.mark_in_flight(req.prompt_uid, -1)

        return pipeline

    async def _dispatch_trajectory(
        self,
        ctx: ProduceContext,
        req: PromptRequest,
        traj: RolloutState,
    ) -> None:
        if traj.status == Status.COMPLETED:
            state, group = await self._aggregator.add_trajectory(traj)
            if state is None:
                logger.debug(
                    f"[{ctx.task_name}] dispatch orphan: prompt_uid={req.prompt_uid}, "
                    f"traj_uid={traj.uid} (aggregation already finalized or dropped)"
                )
                # The trajectory never entered an aggregation, so plasma
                # refs (routed_experts / pixel_values) won't be reached by
                # any later READY / STOPPED cleanup. Free them here.
                free_rollout_state_refs(traj)
                return
            logger.debug(
                f"[{ctx.task_name}] dispatch COMPLETED: prompt_uid={req.prompt_uid}, "
                f"traj_uid={traj.uid} -> {state.name}"
                + (f" (group_size={len(group)})" if group is not None else "")
            )
            if state is GroupState.READY:
                assert group is not None
                batch_judger = self._resolve_batch_judger(group[0])
                if batch_judger is not None:
                    judged = await batch_judger.judge(group)
                    group = judged
                await ctx.put_generated_group(group)
            elif state is GroupState.NEEDS_MORE:
                # All-equal rewards after min_repeat: extend the aggregation by
                # spawning up to ``min_repeat`` more trajectories in parallel,
                # capped at the headroom reported by the policy so the total
                # commitment (completed + in_flight) never exceeds
                # ``max_repeat``. When headroom hits zero we skip the submit
                # entirely and let the remaining in-flight drain; the policy
                # will emit STOPPED once ``len(completed) >= max_repeat``.
                ctx.progress.mark_needs_more(ctx.task_name)
                snapshot = await self._aggregator.get_snapshot(req.prompt_uid)
                if snapshot is None:
                    # Aggregation was dropped concurrently; nothing to resubmit.
                    return
                headroom = self._policy.should_spawn_more(snapshot)
                if headroom <= 0:
                    logger.debug(
                        f"[{ctx.task_name}] NEEDS_MORE: prompt_uid={req.prompt_uid} "
                        f"commitment already at cap; waiting for in-flight to drain"
                    )
                    return
                n_to_submit = min(self._policy.min_repeat, headroom)
                logger.debug(
                    f"[{ctx.task_name}] NEEDS_MORE: prompt_uid={req.prompt_uid} "
                    f"submitting {n_to_submit} more trajectories (headroom={headroom})"
                )
                # NEEDS_MORE retries sit below COLLECTING / PARTIAL_RESUME so
                # the first round of any new prompt's min_repeat fills before
                # we re-roll an all-equal-reward group.
                req.priority = PRIORITY_NEEDS_MORE
                for _ in range(n_to_submit):
                    await self._scheduler.submit_front(req)
            elif state is GroupState.COLLECTING:
                # No re-submit: the ``min_repeat`` initial spawns queued in
                # _preload_prompts keep this prompt saturated until it
                # transitions to READY or NEEDS_MORE.
                pass
            elif state is GroupState.STOPPED:
                await self._aggregator.drop(req.prompt_uid)
                ctx.progress.mark_stopped(ctx.task_name)
            return

        if traj.status == Status.ABORTED:
            # logger.info(
            #     f"[{ctx.task_name}] dispatch ABORTED: prompt_uid={req.prompt_uid}, "
            #     f"traj_uid={traj.uid}, partial_rollout={self._scheduler.config.enable_partial_rollout}"
            # )
            if self._scheduler.config.enable_partial_rollout:
                # The aggregator owns the trajectory's plasma refs once
                # push_pending_keep accepts it; it will free them in drop().
                await self._aggregator.push_pending_keep(traj)
                # Resume of a partial trajectory is the highest-value reuse;
                # keep it ahead of fresh siblings and NEEDS_MORE retries.
                req.priority = PRIORITY_PARTIAL_RESUME
            else:
                # Without partial rollout the ABORTED trajectory is
                # discarded outright, so its routed_experts / pixel_values
                # ObjectRefs would otherwise stay pinned in plasma.
                free_rollout_state_refs(traj)
                # Equivalent to a COLLECTING fresh sibling: same prompt
                # still owes min_repeat completed trajectories.
                req.priority = PRIORITY_COLLECTING
            await self._scheduler.submit_front(req)
            return

        logger.warning(
            f"[{ctx.task_name}] dispatch {traj.status.name}: prompt_uid={req.prompt_uid}, "
            f"traj_uid={traj.uid}; dropping trajectory only."
        )
        # Free the failed trajectory's plasma refs but keep the surrounding
        # aggregation alive. Sibling completed / pending_keep trajectories
        # of the same prompt are still valid; the cross-step
        # _resubmit_unfinished_aggregations path (or in-flight siblings of
        # this round) will top the aggregation back up to min_repeat /
        # NEEDS_MORE-target. Dropping the whole aggregation here would
        # waste already-completed trajectories for one peer's failure.
        free_rollout_state_refs(traj)


# Backward-compatibility aliases. Phase 2 collapses Sync / Async strategies
# into a single trajectory-based class; external code that still imports the
# legacy names (isinstance checks, __all__ re-exports) keeps working.
AsyncProduceStrategy = TrajectoryProduceStrategy
SyncProduceStrategy = TrajectoryProduceStrategy


class _PendingTasks:
    """(deprecated) Concurrency primitive from the removed AsyncProduceStrategy.

    Retained as a standalone shell so tests that imported it at module top
    level continue to import cleanly. Production code no longer references
    it; prefer :class:`TrajectoryScheduler` for new work.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def count(self) -> int:
        return len(self._tasks)

    async def claim_ready(self) -> set[asyncio.Task]:
        async with self._lock:
            ready = {task for task in self._tasks if task.done()}
            self._tasks.difference_update(ready)
            return ready

    async def wait_and_claim(self, *, timeout_s: float) -> set[asyncio.Task]:
        async with self._lock:
            snapshot = set(self._tasks)
        if not snapshot:
            return set()
        done, _ = await asyncio.wait(snapshot, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        async with self._lock:
            claimed = done & self._tasks
            self._tasks.difference_update(claimed)
            return claimed

    async def schedule_one(
        self,
        *,
        max_pending: int,
        should_abort: Callable[[], bool],
        spawn_one: Callable[[], Awaitable[asyncio.Task]],
    ) -> bool:
        async with self._lock:
            if should_abort() or len(self._tasks) >= max_pending:
                return False
            self._tasks.add(await spawn_one())
            return True

    async def _claim_all(self) -> set[asyncio.Task]:
        async with self._lock:
            claimed = set(self._tasks)
            self._tasks.clear()
            return claimed

    async def cancel_all(self) -> int:
        tasks = await self._claim_all()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)
