import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop_manager import (
    ProduceContext,
    ProduceProgress,
    SamplerConfig,
    SyncProduceStrategyConfig,
)
from xtuner.v1.rl.agent_loop_manager.producer import _PendingTasks
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig


class MockRolloutState:
    def __init__(self, id, seq_staleness=1, status=Status.COMPLETED):
        self.id = id
        self.uid = id
        self.status = status
        self.seq_staleness = seq_staleness
        self.response_ids = []
        self.extra_fields = {}


class TestProducer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 1. 模拟 DataloaderConfig 和 Dataloader
        self.mock_dataloader_cfg = MagicMock()
        self.mock_dataloader = MagicMock()
        # 模拟 next(dataloader_iter) 返回 [RolloutState]
        self.mock_dataloader.__iter__.return_value = iter([[MockRolloutState(i)] for i in range(100)])
        self.mock_dataloader_cfg.build.return_value = self.mock_dataloader

        # 2. 模拟 Tokenizer
        self.mock_tokenizer = MagicMock()

        # 3. 准备 ReplayBuffer
        replay_buffer_cfg = AsyncReplayBufferConfig()
        self.replay_buffer = replay_buffer_cfg.build()

    def _build_sampler(self):
        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        return sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)

    def _build_progress(
        self,
        task_name: str,
        target: int,
        train_step: int = 0,
        consumed: int = 0,
    ) -> ProduceProgress:
        return ProduceProgress(
            next_consumer_step=train_step,
            producer_future_step=train_step,
            consumed_samples={task_name: consumed},
            target_samples={task_name: target},
            target_upto_future_step=train_step,
        )

    def _build_agent_loop(self, sleep_by_id: dict[int, float] | None = None):
        mock_agent_loop = MagicMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})

        sleep_by_id = sleep_by_id or {}

        async def mock_gen(rs, **kwargs):
            await asyncio.sleep(sleep_by_id.get(rs[0].id, 0.0))
            for r in rs:
                r.seq_staleness = kwargs.get("model_step", kwargs.get("train_step", 0))
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop.generate_group = mock_gen
        return mock_agent_loop

    def _build_context(
        self,
        strategy,
        task_name: str,
        agent_loop,
        sampler,
        *,
        batch_size: int,
        train_step: int = 0,
        model_step: int = 0,
        progress: ProduceProgress | None = None,
        update_event: asyncio.Event | None = None,
    ) -> ProduceContext:
        # 测试只走新的 ProduceContext 入口，不再覆盖旧散装参数兼容逻辑。
        if progress is None:
            progress = self._build_progress(task_name, target=batch_size, train_step=train_step)
        if update_event is None:
            update_event = asyncio.Event()
        return ProduceContext(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=self.replay_buffer,
            task_batch_size=batch_size,
            task_name=task_name,
            train_step=train_step,
            update_event=update_event,
            model_step=model_step,
            progress=progress,
            is_valid_sample_fn=strategy.is_valid_sample_fn,
            stale_threshold=getattr(strategy, "stale_threshold", None),
        )

    def test_produce_progress_methods_keep_absolute_window(self):
        progress = ProduceProgress.build(["task_a", "task_b"])

        def allocate(batch_size: int, step: int) -> dict[str, int]:
            self.assertEqual(batch_size, 4)
            return {"task_a": step, "task_b": batch_size - step}

        current_sizes = progress.ensure_target_upto(
            batch_size=4,
            future_step=2,
            allocate_batch_sizes=allocate,
        )

        self.assertEqual(current_sizes, {"task_a": 2, "task_b": 2})
        self.assertEqual(progress.target_samples, {"task_a": 3, "task_b": 5})
        self.assertEqual(progress.target_upto_future_step, 2)

        progress.begin_consume(2)
        progress.mark_consumed({"task_a": 1, "task_b": 2})
        progress.finish_consume(2)
        progress.advance_future_step()
        self.assertEqual(progress.next_consumer_step, 3)
        self.assertEqual(progress.producer_future_step, 2)
        self.assertEqual(progress.consumed_samples, {"task_a": 1, "task_b": 2})

        local_progress = ProduceProgress.build_local(["task_a", "task_b"], {"task_a": 1, "task_b": 3}, 7)
        self.assertEqual(local_progress.target_samples, {"task_a": 1, "task_b": 3})
        self.assertEqual(progress.target_samples, {"task_a": 3, "task_b": 5})

        consumed_ref = progress.consumed_samples
        target_ref = progress.target_samples
        progress.load_state_dict(
            {
                "next_consumer_step": 8,
                "producer_future_step": 9,
                "consumed_samples": {"task_a": 4, "task_b": 5},
                "target_samples": {"task_a": 6, "task_b": 7},
                "target_upto_future_step": 10,
            }
        )
        self.assertIs(progress.consumed_samples, consumed_ref)
        self.assertIs(progress.target_samples, target_ref)
        self.assertEqual(progress.state_dict()["target_samples"], {"task_a": 6, "task_b": 7})

    async def test_pending_tasks_claim_ready_only_once(self):
        pending_tasks = _PendingTasks()

        async def spawn_one():
            async def done():
                return "done"

            return asyncio.create_task(done())

        scheduled = await pending_tasks.schedule_one(
            max_pending=1,
            should_abort=lambda: False,
            spawn_one=spawn_one,
        )
        self.assertTrue(scheduled)
        self.assertEqual(pending_tasks.count(), 1)

        await asyncio.sleep(0)
        claimed = await pending_tasks.claim_ready()
        self.assertEqual(len(claimed), 1)
        self.assertEqual(await pending_tasks.claim_ready(), set())
        self.assertEqual(pending_tasks.count(), 0)

    async def test_pending_tasks_schedule_respects_abort_and_limit(self):
        pending_tasks = _PendingTasks()
        spawn_count = 0

        async def spawn_one():
            nonlocal spawn_count
            spawn_count += 1

            async def wait_forever():
                await asyncio.Event().wait()

            return asyncio.create_task(wait_forever())

        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=0, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: True, spawn_one=spawn_one)
        )
        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertEqual(spawn_count, 1)

        self.assertEqual(await pending_tasks.cancel_all(), 1)
        self.assertEqual(pending_tasks.count(), 0)

    async def test_pending_tasks_cancel_all_clears_before_wait_claims(self):
        pending_tasks = _PendingTasks()

        async def spawn_one():
            async def wait_forever():
                await asyncio.Event().wait()

            return asyncio.create_task(wait_forever())

        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertEqual(await pending_tasks.cancel_all(), 1)
        self.assertEqual(await pending_tasks.wait_and_claim(timeout_s=0), set())
        self.assertEqual(pending_tasks.count(), 0)

    async def test_sampler_with_replay_buffer(self):
        task_name = "test_task"
        sampler = self._build_sampler()

        # 场景 A: ReplayBuffer 为空，从 Dataloader 拿
        data = await sampler.sample(task_name)
        self.assertEqual(data[0].id, 0)

        # 场景 B: ReplayBuffer 有多个候选状态，按列表顺序优先拿
        aborted_item = MockRolloutState(999, status=Status.ABORTED)
        expired_item = MockRolloutState(1000, status=Status.EXPIRED)
        await self.replay_buffer.put([aborted_item], task_name)
        await self.replay_buffer.put([expired_item], task_name)

        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].id, 1000)

        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].id, 999)

        # 场景 C: ReplayBuffer 对应状态都为空，回退到 Dataloader
        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].id, 1)

    async def test_put_generated_group_only_validates_completed_group(self):
        task_name = "test_valid_completed_only"
        valid_checked_statuses = []

        def is_valid_sample_fn(samples):
            valid_checked_statuses.append([sample.status for sample in samples])
            return False

        strategy = SyncProduceStrategyConfig(is_valid_sample_fn=is_valid_sample_fn).build()
        ctx = self._build_context(
            strategy,
            task_name,
            self._build_agent_loop(),
            self._build_sampler(),
            batch_size=1,
        )

        completed_group = [MockRolloutState(1, status=Status.COMPLETED)]
        self.assertFalse(await ctx.put_generated_group(completed_group))
        self.assertEqual(completed_group[0].status, Status.FILTERED)

        aborted_group = [MockRolloutState(2, status=Status.ABORTED)]
        self.assertFalse(await ctx.put_generated_group(aborted_group))
        self.assertEqual(aborted_group[0].status, Status.ABORTED)

        self.assertEqual(valid_checked_statuses, [[Status.COMPLETED]])
        self.assertEqual(await self.replay_buffer.count(task_name, Status.FILTERED), 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.ABORTED), 1)

    async def test_refresh_staleness_refreshes_before_expire_check(self):
        task_name = "test_refresh_leftover"
        stale_item = MockRolloutState(1000, seq_staleness=0, status=Status.COMPLETED)
        stale_item.response_model_steps = [3]
        await self.replay_buffer.put([stale_item], task_name)

        expired_counts = await self.replay_buffer.refresh_staleness(
            task_stale_thresholds={task_name: 2},
            current_train_step=6,
        )
        expired_groups = await self.replay_buffer.get(10, task_name, Status.EXPIRED)

        self.assertEqual(expired_counts, {task_name: 1})
        self.assertEqual(len(expired_groups), 1)
        self.assertEqual(expired_groups[0][0].seq_staleness, 2)
