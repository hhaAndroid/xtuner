"""Phase 3 tests: progressive sampling config, telemetry, count_trajectories, count_unit plumbing."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop_manager import (
    GroupPolicyConfig,
    ProduceContext,
    ProduceProgress,
    ProgressiveProduceStrategyConfig,
    PromptRequest,
    SyncProduceStrategyConfig,
    TrajectorySchedulerConfig,
)
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig


def _make_real_prompt(message_uid: int) -> RolloutState:
    return RolloutState(
        message=[{"role": "user", "content": "q"}],
        message_uid=message_uid,
    )


def _make_traj(message_uid: int, uid: int, score: float) -> RolloutState:
    return RolloutState(
        message=[],
        message_uid=message_uid,
        uid=uid,
        reward={"score": score},
        status=Status.COMPLETED,
    )


class _MockState:
    def __init__(self, uid: int, status: Status = Status.COMPLETED, staleness: int = 0):
        self.uid = uid
        self.status = status
        self.seq_staleness = staleness
        self.response_ids: list[int] = []
        self.response_model_steps: list[int] | None = None


class TestProgressiveProduceStrategyConfig(unittest.TestCase):
    def _make_cfg(self, min_r: int = 2, max_r: int = 4, **policy_kwargs):
        return ProgressiveProduceStrategyConfig(
            group_policy=GroupPolicyConfig(
                min_repeat=min_r, max_repeat=max_r, **policy_kwargs
            ),
            scheduler=TrajectorySchedulerConfig(max_on_fly=8),
        )

    def test_build_sets_count_unit_trajectories(self):
        strategy = self._make_cfg().build()
        self.assertEqual(strategy.count_unit, "trajectories")

    def test_policy_matches_config(self):
        strategy = self._make_cfg(min_r=2, max_r=6).build()
        self.assertEqual(strategy.aggregator.policy.min_repeat, 2)
        self.assertEqual(strategy.aggregator.policy.max_repeat, 6)

    def test_stop_when_all_equal_true_by_default(self):
        strategy = self._make_cfg().build()
        self.assertTrue(strategy.aggregator.policy.config.stop_when_all_equal)

    def test_stop_when_all_equal_explicit_false(self):
        strategy = self._make_cfg(stop_when_all_equal=False).build()
        self.assertFalse(strategy.aggregator.policy.config.stop_when_all_equal)

    def test_prompt_repeat_k_is_ignored(self):
        strategy = self._make_cfg(min_r=3, max_r=5).build(prompt_repeat_k=99)
        self.assertEqual(strategy.aggregator.policy.min_repeat, 3)
        self.assertEqual(strategy.aggregator.policy.max_repeat, 5)

    def test_legacy_shim_keeps_groups_unit(self):
        strategy = SyncProduceStrategyConfig().build(prompt_repeat_k=4)
        self.assertEqual(strategy.count_unit, "groups")


class TestProduceProgressTelemetry(unittest.TestCase):
    def test_build_initializes_telemetry_dicts(self):
        progress = ProduceProgress.build(["t1", "t2"])
        self.assertEqual(progress.stopped_prompts, {"t1": 0, "t2": 0})
        self.assertEqual(progress.needs_more_reentries, {"t1": 0, "t2": 0})

    def test_build_local_initializes_telemetry_dicts(self):
        progress = ProduceProgress.build_local(["t"], {"t": 4}, train_step=0)
        self.assertEqual(progress.stopped_prompts, {"t": 0})
        self.assertEqual(progress.needs_more_reentries, {"t": 0})

    def test_mark_stopped_accumulates(self):
        progress = ProduceProgress.build(["t1", "t2"])
        progress.mark_stopped("t1")
        progress.mark_stopped("t1", 4)
        progress.mark_stopped("t2", 2)
        self.assertEqual(progress.stopped_prompts, {"t1": 5, "t2": 2})

    def test_mark_needs_more_accumulates(self):
        progress = ProduceProgress.build(["t"])
        progress.mark_needs_more("t", 3)
        progress.mark_needs_more("t")
        self.assertEqual(progress.needs_more_reentries, {"t": 4})

    def test_state_dict_roundtrip_preserves_telemetry(self):
        progress = ProduceProgress.build(["t"])
        progress.mark_stopped("t", 5)
        progress.mark_needs_more("t", 10)

        state = progress.state_dict()
        fresh = ProduceProgress.build(["t"])
        fresh.load_state_dict(state)

        self.assertEqual(fresh.stopped_prompts, {"t": 5})
        self.assertEqual(fresh.needs_more_reentries, {"t": 10})

    def test_load_state_dict_backward_compatible(self):
        fresh = ProduceProgress.build(["t"])
        # A pre-Phase-3 state_dict had no stopped_prompts / needs_more_reentries keys.
        fresh.load_state_dict(
            {
                "next_consumer_step": 0,
                "producer_future_step": 0,
                "consumed_samples": {"t": 0},
                "target_samples": {"t": 0},
                "target_upto_future_step": 0,
            }
        )
        self.assertEqual(fresh.stopped_prompts, {})
        self.assertEqual(fresh.needs_more_reentries, {})


class TestReplayBufferCountTrajectories(unittest.IsolatedAsyncioTestCase):
    async def test_sums_group_sizes(self):
        rb = AsyncReplayBufferConfig().build()
        await rb.put([_MockState(i) for i in range(3)], "t")
        await rb.put([_MockState(i + 10) for i in range(2)], "t")
        count = await rb.count_trajectories("t", Status.COMPLETED)
        self.assertEqual(count, 5)

    async def test_zero_when_no_match(self):
        rb = AsyncReplayBufferConfig().build()
        count = await rb.count_trajectories("t", Status.COMPLETED)
        self.assertEqual(count, 0)

    async def test_status_filter(self):
        rb = AsyncReplayBufferConfig().build()
        await rb.put([_MockState(1, status=Status.FAILED)], "t")
        await rb.put([_MockState(2, status=Status.COMPLETED)], "t")
        self.assertEqual(await rb.count_trajectories("t", Status.COMPLETED), 1)
        self.assertEqual(await rb.count_trajectories("t", Status.FAILED), 1)

    async def test_task_isolation(self):
        rb = AsyncReplayBufferConfig().build()
        await rb.put([_MockState(1)], "t1")
        await rb.put([_MockState(2), _MockState(3)], "t2")
        self.assertEqual(await rb.count_trajectories("t1", Status.COMPLETED), 1)
        self.assertEqual(await rb.count_trajectories("t2", Status.COMPLETED), 2)


class TestStrategyDispatchTelemetry(unittest.IsolatedAsyncioTestCase):
    def _build_strategy(self, min_r: int = 2, max_r: int = 2):
        cfg = ProgressiveProduceStrategyConfig(
            group_policy=GroupPolicyConfig(
                min_repeat=min_r, max_repeat=max_r, stop_when_all_equal=True
            ),
            scheduler=TrajectorySchedulerConfig(max_on_fly=4),
        )
        return cfg.build()

    def _build_ctx(self, strategy, progress: ProduceProgress, task_name: str) -> ProduceContext:
        replay_buffer = AsyncReplayBufferConfig().build()
        agent_loop = MagicMock()
        sampler = MagicMock()
        sampler.sample_prompt = AsyncMock()
        return ProduceContext(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=replay_buffer,
            task_batch_size=1,
            task_name=task_name,
            train_step=0,
            update_event=asyncio.Event(),
            model_step=0,
            progress=progress,
            is_valid_sample_fn=strategy.is_valid_sample_fn,
        )

    async def test_stopped_counter_ticks_when_max_reached(self):
        strategy = self._build_strategy(min_r=2, max_r=2)
        prompt = _make_real_prompt(42)
        await strategy.aggregator.register_prompt(prompt, "t")
        progress = ProduceProgress.build(["t"])
        ctx = self._build_ctx(strategy, progress, "t")
        req = PromptRequest(prompt_uid=42, task_name="t", prompt=prompt)

        await strategy._dispatch_trajectory(ctx, req, _make_traj(42, uid=101, score=0.5))
        # First completion: COLLECTING (n=1 < min=2). No telemetry change.
        self.assertEqual(progress.stopped_prompts["t"], 0)
        self.assertEqual(progress.needs_more_reentries["t"], 0)

        await strategy._dispatch_trajectory(ctx, req, _make_traj(42, uid=102, score=0.5))
        # Second completion: n=2, all-equal, no headroom (max=2) -> STOPPED.
        self.assertEqual(progress.stopped_prompts["t"], 1)
        self.assertFalse(await strategy.aggregator.exists(42))

    async def test_needs_more_counter_ticks_when_headroom_remains(self):
        strategy = self._build_strategy(min_r=2, max_r=4)
        prompt = _make_real_prompt(42)
        await strategy.aggregator.register_prompt(prompt, "t")
        progress = ProduceProgress.build(["t"])
        ctx = self._build_ctx(strategy, progress, "t")
        req = PromptRequest(prompt_uid=42, task_name="t", prompt=prompt)

        await strategy._dispatch_trajectory(ctx, req, _make_traj(42, uid=101, score=0.5))
        await strategy._dispatch_trajectory(ctx, req, _make_traj(42, uid=102, score=0.5))
        # Both identical rewards, n=2, max=4 -> NEEDS_MORE (headroom = 2).
        self.assertEqual(progress.needs_more_reentries["t"], 1)
        self.assertTrue(await strategy.aggregator.exists(42))

    async def test_ready_writes_group_to_buffer_and_leaves_counters_untouched(self):
        strategy = self._build_strategy(min_r=2, max_r=4)
        prompt = _make_real_prompt(42)
        await strategy.aggregator.register_prompt(prompt, "t")
        progress = ProduceProgress.build(["t"])
        ctx = self._build_ctx(strategy, progress, "t")
        req = PromptRequest(prompt_uid=42, task_name="t", prompt=prompt)

        await strategy._dispatch_trajectory(ctx, req, _make_traj(42, uid=101, score=0.0))
        await strategy._dispatch_trajectory(ctx, req, _make_traj(42, uid=102, score=1.0))
        # Rewards differ at n=2 -> READY; aggregator finalized.
        self.assertEqual(progress.stopped_prompts["t"], 0)
        self.assertEqual(progress.needs_more_reentries["t"], 0)
        self.assertFalse(await strategy.aggregator.exists(42))
        groups = await ctx.replay_buffer.get(10, "t", Status.COMPLETED)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)


if __name__ == "__main__":
    unittest.main()
