import asyncio
import copy
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable
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
from xtuner.v1.rl.agent_loop_manager.trajectory_scheduler import (
    PromptRequest,
    TrajectoryScheduler,
    TrajectorySchedulerConfig,
)
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout.utils import pause_generation
from xtuner.v1.rl.utils import calculate_seq_staleness, create_task
from xtuner.v1.utils import get_logger

from .sampler import Sampler


logger = get_logger()
GROUP_GENERATE_TIME_KEY = "group_generate_time_s"


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
    - consumed_samples：各 task 已被 consumer 从 replay buffer 取走的 group 绝对累计数。
    - target_samples：各 task 截至 target_upto_future_step 应生产出的 group 绝对累计目标。
    - target_upto_future_step：target_samples 已覆盖到的最大 future step。
    """

    next_consumer_step: int = 1
    producer_future_step: int = 1
    consumed_samples: dict[str, int] = field(default_factory=dict)
    target_samples: dict[str, int] = field(default_factory=dict)
    target_upto_future_step: int = 0

    @classmethod
    def build(cls, task_names: list[str]) -> "ProduceProgress":
        return cls(
            consumed_samples={task_name: 0 for task_name in task_names},
            target_samples={task_name: 0 for task_name in task_names},
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
        is_completed = get_group_status(group) == Status.COMPLETED
        if is_completed:
            is_valid = self.is_valid_sample_fn(group)
            if not is_valid:
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
        is_completed = get_group_status(group) == Status.COMPLETED
        return is_completed


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
       aggregation is dropped and a stats counter ticks.

    Phase 2 wires this strategy through legacy config shims with
    ``min_repeat == max_repeat == prompt_repeat_k`` and
    ``stop_when_all_equal=False`` so the emitted groups match the legacy
    AsyncProduceStrategy byte-for-byte modulo trajectory arrival order.

    Args:
        scheduler (TrajectoryScheduler): Global concurrency / queue manager.
        group_policy (GroupPolicy): Policy driving aggregation decisions.
        over_sample_threshold (float): Extra prompts kept preloaded beyond
            the strict target, mirroring the legacy knob.
        is_valid_sample_fn (IsValidSampleFn): Filter applied on finalized
            groups before they are written to the replay buffer.
        should_continue_fn (ShouldContinueFn): Loop termination predicate
            evaluated against ``ctx.available_count()``.
    """

    def __init__(
        self,
        *,
        scheduler: TrajectoryScheduler,
        group_policy: GroupPolicy,
        over_sample_threshold: float,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ) -> None:
        super().__init__(is_valid_sample_fn, should_continue_fn)
        self._scheduler = scheduler
        self._policy = group_policy
        self._aggregator = GroupAggregator(group_policy)
        self._over_sample_threshold = over_sample_threshold
        self._stopped_count: int = 0
        self._needs_more_count: int = 0

    @property
    def stale_threshold(self) -> int:
        return self._scheduler.stale_threshold

    @property
    def aggregator(self) -> GroupAggregator:
        return self._aggregator

    @property
    def scheduler(self) -> TrajectoryScheduler:
        return self._scheduler

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return self._scheduler.is_model_expired(train_step, model_step)

    def pending_task_count(self) -> int:
        return self._scheduler.pending_count()

    async def pause_produce(self, ctx: ProduceContext) -> float:
        pause_start = time.perf_counter()
        if self._scheduler.pending_count() == 0:
            return 0.0
        rollout_ctl = await get_agent_loop_rollout_ctl(ctx.agent_loop)
        await pause_generation(rollout_ctl)
        await self._scheduler.pause_and_cleanup()
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

        logger.info(
            f"Starting produce_batch for task {ctx.task_name}: "
            f"target_abs={ctx.target_abs}, over_sample_threshold={self._over_sample_threshold}, "
            f"max_on_fly={self._scheduler.config.max_on_fly}."
        )

        runner = self._build_runner(ctx)

        while True:
            if ctx.should_abort():
                return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
            if self.is_model_expired(ctx.train_step, ctx.model_step):
                return ProduceBatchStatus.EXPIRED_BATCH

            available = await ctx.available_count()
            if not self.should_continue_fn(available, ctx.target_abs):
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
                        f"Produce stalled for task {ctx.task_name}: no pending tasks, "
                        "empty queue, no active aggregation."
                    )
                    return ProduceBatchStatus.NORMAL

            await self._scheduler.wait_first_completed(timeout_s=1.0)

        if self._scheduler.config.wait_until_all_ready:
            await self._scheduler.drain()

        return ProduceBatchStatus.NORMAL

    async def state_dict(self) -> dict[str, Any]:
        return {
            "aggregator": await self._aggregator.state_dict(),
            "stopped_count": self._stopped_count,
            "needs_more_count": self._needs_more_count,
        }

    async def load_state_dict(self, state: dict[str, Any]) -> None:
        if "aggregator" in state:
            await self._aggregator.load_state_dict(state["aggregator"])
        self._stopped_count = int(state.get("stopped_count", 0))
        self._needs_more_count = int(state.get("needs_more_count", 0))

    async def _preload_prompts(self, ctx: ProduceContext, available: int) -> None:
        groups_needed = ctx.target_abs - available
        if groups_needed <= 0:
            return
        active_prompts = await self._aggregator.active_count() + self._scheduler.queue_len()
        oversample = math.ceil(self._over_sample_threshold * ctx.task_batch_size)
        target_active = groups_needed + oversample
        to_preload = max(0, target_active - active_prompts)
        for _ in range(to_preload):
            if ctx.should_abort():
                return
            prompt_req = await ctx.sampler.sample_prompt(task_name=ctx.task_name)
            await self._aggregator.register_prompt(prompt_req.prompt, ctx.task_name)
            await self._scheduler.submit(prompt_req)

    def _build_runner(self, ctx: ProduceContext) -> Callable[[PromptRequest], Awaitable[None]]:
        async def runner(req: PromptRequest) -> None:
            try:
                await self._aggregator.mark_in_flight(req.prompt_uid, +1)
                pending: RolloutState | None = None
                if self._scheduler.config.enable_partial_rollout:
                    pending = await self._aggregator.pop_pending_keep(req.prompt_uid)
                if pending is not None:
                    input_state = pending
                else:
                    input_state = copy.deepcopy(req.prompt)
                    input_state.uid = uuid4().int
                    input_state.session_uid = input_state.uid
                result_list = await ctx.generate_group(
                    [input_state],
                    enable_partial_rollout=self._scheduler.config.enable_partial_rollout,
                )
                traj = result_list[0]
                await self._dispatch_trajectory(ctx, req, traj)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - must not break scheduler state
                logger.error(
                    f"Trajectory runner failed for prompt_uid={req.prompt_uid}: "
                    f"{type(exc).__name__}: {exc}",
                    exc_info=exc,
                )
            finally:
                await self._aggregator.mark_in_flight(req.prompt_uid, -1)

        return runner

    async def _dispatch_trajectory(
        self,
        ctx: ProduceContext,
        req: PromptRequest,
        traj: RolloutState,
    ) -> None:
        if traj.status == Status.COMPLETED:
            state, group = await self._aggregator.add_trajectory(traj)
            if state is None:
                return
            if state is GroupState.READY:
                assert group is not None
                await ctx.put_generated_group(group)
            elif state is GroupState.NEEDS_MORE:
                self._needs_more_count += 1
                await self._scheduler.submit_front(req)
            elif state is GroupState.COLLECTING:
                await self._scheduler.submit(req)
            elif state is GroupState.STOPPED:
                await self._aggregator.drop(req.prompt_uid)
                self._stopped_count += 1
            return

        if traj.status == Status.ABORTED:
            if self._scheduler.config.enable_partial_rollout:
                await self._aggregator.push_pending_keep(traj)
            await self._scheduler.submit_front(req)
            return

        logger.warning(
            f"Dropping aggregation prompt_uid={req.prompt_uid} because trajectory "
            f"uid={traj.uid} ended with status {traj.status}."
        )
        await self._aggregator.drop(req.prompt_uid)


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
