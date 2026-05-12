"""Phase 2 regression tests for :class:`TrajectoryProduceStrategy`.

Covers the legacy config shims (``SyncProduceStrategyConfig`` /
``AsyncProduceStrategyConfig``), group-equivalence at varying
``prompt_repeat_k``, early-exit paths (abort, expired, missing progress),
and state-dict round-trip through the aggregator.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop_manager import (
    AsyncProduceStrategyConfig,
    ProduceBatchStatus,
    ProduceContext,
    ProduceProgress,
    SamplerConfig,
    SyncProduceStrategyConfig,
)
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig


class _MockPrompt:
    """Minimal RolloutState duck-type used as dataloader items."""

    def __init__(self) -> None:
        self.message_uid: int | None = None
        self.uid: int | None = None
        self.session_uid: int | None = None
        self.status = Status.INIT
        self.response_ids: list[int] = []
        self.response_model_steps: list[int] | None = None
        self.seq_staleness: int = 0
        self.extra_fields: dict = {}
        self.mm_info = None
        self.reward: dict | None = None


def _make_real_prompt(message_uid: int) -> RolloutState:
    return RolloutState(
        message=[{"role": "user", "content": "q"}],
        message_uid=message_uid,
    )


def _build_dataloader(n: int = 50) -> MagicMock:
    items = [[_MockPrompt()] for _ in range(n)]
    dl = MagicMock()
    dl.__iter__.side_effect = lambda: iter(items)
    dl.set_epoch = MagicMock()
    return dl


def _build_agent_loop() -> MagicMock:
    mock = MagicMock()
    mock.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
    mock.rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
    mock.rollout_ctl.get_rollout_metadata.remote = AsyncMock(
        return_value={"server_url_dict": {}}
    )

    async def generate_group(rollout_states, **kwargs):
        # Yield to the scheduler so wait_first_completed observes progress.
        await asyncio.sleep(0)
        for rollout_state in rollout_states:
            rollout_state.status = Status.COMPLETED
            rollout_state.reward = {"score": float((rollout_state.uid or 0) % 100)}
        return rollout_states

    mock.generate_group = generate_group
    return mock


def _build_progress(task_name: str, *, target: int, train_step: int = 0) -> ProduceProgress:
    return ProduceProgress(
        next_consumer_step=train_step,
        producer_future_step=train_step,
        consumed_samples={task_name: 0},
        target_samples={task_name: target},
        target_upto_future_step=train_step,
    )


def _build_context(
    strategy,
    agent_loop,
    sampler,
    replay_buffer,
    task_name: str,
    *,
    target: int,
    train_step: int = 0,
    model_step: int = 0,
    update_event: asyncio.Event | None = None,
    progress: ProduceProgress | None = None,
) -> ProduceContext:
    return ProduceContext(
        agent_loop=agent_loop,
        sampler=sampler,
        replay_buffer=replay_buffer,
        task_batch_size=target,
        task_name=task_name,
        train_step=train_step,
        update_event=update_event if update_event is not None else asyncio.Event(),
        model_step=model_step,
        progress=progress if progress is not None else _build_progress(task_name, target=target, train_step=train_step),
        is_valid_sample_fn=strategy.is_valid_sample_fn,
        stale_threshold=getattr(strategy, "stale_threshold", None),
    )


class TestLegacyConfigShims(unittest.TestCase):
    def test_sync_config_forwards_prompt_repeat_k(self):
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=4)
        self.assertEqual(strategy.aggregator.policy.min_repeat, 4)
        self.assertEqual(strategy.aggregator.policy.max_repeat, 4)

    def test_sync_config_sets_wait_until_all_ready(self):
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=2)
        self.assertTrue(strategy.scheduler.config.wait_until_all_ready)

    def test_async_config_forwards_scheduler_knobs(self):
        cfg = AsyncProduceStrategyConfig(
            over_sample_threshold=0.5,
            max_staleness=2,
            enable_partial_rollout=True,
            tail_batch_trigger_size=3,
        )
        strategy = cfg.build(sync_weights_interval=10, prompt_repeat_k=3)
        self.assertFalse(strategy.scheduler.config.wait_until_all_ready)
        self.assertEqual(strategy.scheduler.config.max_staleness, 2)
        self.assertTrue(strategy.scheduler.config.enable_partial_rollout)
        self.assertEqual(strategy.scheduler.config.tail_batch_trigger_size, 3)
        # stale_threshold = (max_staleness + 1) * sync_weights_interval
        self.assertEqual(strategy.stale_threshold, 30)

    def test_shims_disable_stop_when_all_equal(self):
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=4)
        self.assertFalse(strategy.aggregator.policy.config.stop_when_all_equal)

    def test_shims_allow_prompt_repeat_k_of_one(self):
        # K=1 is the legacy degenerate case; the shim must not reject it.
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=1)
        self.assertEqual(strategy.aggregator.policy.min_repeat, 1)
        self.assertEqual(strategy.aggregator.policy.max_repeat, 1)


class TestTrajectoryProduceStrategyEquivalence(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mock_dataloader_cfg = MagicMock()
        self.mock_dataloader_cfg.build.return_value = _build_dataloader(n=60)
        self.mock_tokenizer = MagicMock()
        self.replay_buffer = AsyncReplayBufferConfig().build()

    def _build_sampler(self):
        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        return sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)

    async def test_k_equals_four_produces_groups_of_four(self):
        task_name = "t"
        sampler = self._build_sampler()
        agent_loop = _build_agent_loop()
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=4)
        ctx = _build_context(
            strategy, agent_loop, sampler, self.replay_buffer, task_name, target=2
        )
        status = await strategy.produce_batch(ctx)
        self.assertEqual(status, ProduceBatchStatus.NORMAL)

        groups = await self.replay_buffer.get(50, task_name, Status.COMPLETED)
        self.assertGreaterEqual(len(groups), 2)
        for group in groups:
            self.assertEqual(len(group), 4)

    async def test_k_equals_one_produces_singleton_groups(self):
        task_name = "t"
        sampler = self._build_sampler()
        agent_loop = _build_agent_loop()
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=1)
        ctx = _build_context(
            strategy, agent_loop, sampler, self.replay_buffer, task_name, target=3
        )
        status = await strategy.produce_batch(ctx)
        self.assertEqual(status, ProduceBatchStatus.NORMAL)

        groups = await self.replay_buffer.get(50, task_name, Status.COMPLETED)
        self.assertGreaterEqual(len(groups), 3)
        for group in groups:
            self.assertEqual(len(group), 1)


class TestEarlyExitPaths(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mock_dataloader_cfg = MagicMock()
        self.mock_dataloader_cfg.build.return_value = _build_dataloader(n=20)
        self.mock_tokenizer = MagicMock()
        self.replay_buffer = AsyncReplayBufferConfig().build()

    def _build_sampler(self):
        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        return sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)

    async def test_abort_flag_returns_update_weight_without_sampling(self):
        task_name = "t"
        sampler = self._build_sampler()
        agent_loop = _build_agent_loop()
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=2)
        update_event = asyncio.Event()
        update_event.set()
        ctx = _build_context(
            strategy,
            agent_loop,
            sampler,
            self.replay_buffer,
            task_name,
            target=2,
            update_event=update_event,
        )
        status = await strategy.produce_batch(ctx)
        self.assertEqual(status, ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT)
        # No prompts pulled from the dataloader when aborting immediately.
        self.assertEqual(sampler._consumed_samples, 0)

    async def test_stale_model_returns_expired_batch(self):
        task_name = "t"
        sampler = self._build_sampler()
        agent_loop = _build_agent_loop()
        cfg = AsyncProduceStrategyConfig(max_staleness=1)
        strategy = cfg.build(sync_weights_interval=10, prompt_repeat_k=2)
        # stale_threshold = (1+1)*10 = 20; train=100, model=0 -> staleness=100 -> expired.
        ctx = _build_context(
            strategy,
            agent_loop,
            sampler,
            self.replay_buffer,
            task_name,
            target=2,
            train_step=100,
            model_step=0,
        )
        status = await strategy.produce_batch(ctx)
        self.assertEqual(status, ProduceBatchStatus.EXPIRED_BATCH)

    async def test_missing_task_progress_raises(self):
        sampler = self._build_sampler()
        agent_loop = _build_agent_loop()
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=2)
        progress = ProduceProgress(
            next_consumer_step=0,
            producer_future_step=0,
            consumed_samples={"other": 0},
            target_samples={"other": 2},
            target_upto_future_step=0,
        )
        ctx = _build_context(
            strategy,
            agent_loop,
            sampler,
            self.replay_buffer,
            "t",
            target=2,
            progress=progress,
        )
        with self.assertRaises(KeyError):
            await strategy.produce_batch(ctx)


class TestStateDictRoundTrip(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_aggregator_state(self):
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=2)
        await strategy.aggregator.register_prompt(_make_real_prompt(42), "t")

        state = await strategy.state_dict()

        new_strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=2)
        await new_strategy.load_state_dict(state)
        self.assertTrue(await new_strategy.aggregator.exists(42))

    async def test_empty_state_dict_noop_restore(self):
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=2)
        await strategy.load_state_dict({})
        self.assertEqual(await strategy.aggregator.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
