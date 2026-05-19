"""Unit tests for :mod:`xtuner.v1.rl.agent_loop_manager.trajectory_scheduler`."""

import asyncio
import unittest

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.agent_loop_manager.trajectory_scheduler import (
    PromptRequest,
    TrajectorySchedulerConfig,
    _PromptDeque,
    calculate_stale_threshold,
)


def _make_req(uid: int, task: str = "t") -> PromptRequest:
    return PromptRequest(
        prompt_uid=uid,
        task_name=task,
        prompt=RolloutState(message=[], message_uid=uid),
    )


class TestPromptDeque(unittest.TestCase):
    def test_push_back_pop_order(self):
        d = _PromptDeque()
        d.push_back(_make_req(1))
        d.push_back(_make_req(2))
        d.push_back(_make_req(3))
        self.assertEqual(d.pop().prompt_uid, 1)
        self.assertEqual(d.pop().prompt_uid, 2)
        self.assertEqual(d.pop().prompt_uid, 3)

    def test_push_front_goes_first(self):
        d = _PromptDeque()
        d.push_back(_make_req(1))
        d.push_back(_make_req(2))
        d.push_front(_make_req(99))
        self.assertEqual(d.pop().prompt_uid, 99)
        self.assertEqual(d.pop().prompt_uid, 1)

    def test_len_and_bool(self):
        d = _PromptDeque()
        self.assertEqual(len(d), 0)
        self.assertFalse(d)
        d.push_back(_make_req(1))
        self.assertEqual(len(d), 1)
        self.assertTrue(d)


class TestCalculateStaleThreshold(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(calculate_stale_threshold(0, 10), 10)
        self.assertEqual(calculate_stale_threshold(2, 5), 15)

    def test_negative_staleness_rejected(self):
        with self.assertRaises(ValueError):
            calculate_stale_threshold(-1, 10)

    def test_non_positive_interval_rejected(self):
        with self.assertRaises(ValueError):
            calculate_stale_threshold(1, 0)


class TestSchedulerSpawn(unittest.IsolatedAsyncioTestCase):
    def _build(self, max_on_fly=2, **kwargs):
        cfg = TrajectorySchedulerConfig(max_on_fly=max_on_fly, **kwargs)
        return cfg.build(sync_weights_interval=10)

    async def test_spawn_respects_capacity(self):
        scheduler = self._build(max_on_fly=2)
        release = asyncio.Event()

        async def pipeline(req, release_slot):
            # Hold the slot until the test releases the gate, so we can probe capacity.
            await release.wait()
            release_slot()

        for i in range(4):
            await scheduler.submit(_make_req(i))
        self.assertTrue(await scheduler.spawn_if_slot(pipeline))
        self.assertTrue(await scheduler.spawn_if_slot(pipeline))
        # Third attempt exceeds max_on_fly
        self.assertFalse(await scheduler.spawn_if_slot(pipeline))
        self.assertEqual(scheduler.inflight_count(), 2)
        self.assertEqual(scheduler.pending_count(), 2)
        release.set()
        await scheduler.drain()
        self.assertEqual(scheduler.inflight_count(), 0)
        self.assertEqual(scheduler.pending_count(), 0)

    async def test_spawn_returns_false_when_queue_empty(self):
        scheduler = self._build()

        async def pipeline(req, release_slot):
            release_slot()

        self.assertFalse(await scheduler.spawn_if_slot(pipeline))

    async def test_front_priority_runs_first(self):
        scheduler = self._build(max_on_fly=1)
        seen: list[int] = []

        async def pipeline(req, release_slot):
            seen.append(req.prompt_uid)
            release_slot()

        await scheduler.submit(_make_req(1))
        await scheduler.submit(_make_req(2))
        await scheduler.submit_front(_make_req(99))
        await scheduler.spawn_if_slot(pipeline)
        await scheduler.drain()
        self.assertEqual(seen, [99])
        await scheduler.spawn_if_slot(pipeline)
        await scheduler.drain()
        self.assertEqual(seen, [99, 1])

    async def test_release_slot_unblocks_next_inference(self):
        # Inference releases its slot promptly while the post phase is
        # still running; the next spawn must be admitted while pending_count
        # is still 1 (the post-phase task).
        scheduler = self._build(max_on_fly=1)
        post_gate = asyncio.Event()
        seen: list[int] = []

        async def pipeline(req, release_slot):
            seen.append(req.prompt_uid)
            release_slot()
            await post_gate.wait()  # post phase blocks until the test releases it

        await scheduler.submit(_make_req(1))
        await scheduler.submit(_make_req(2))
        self.assertTrue(await scheduler.spawn_if_slot(pipeline))
        # Post phase of #1 is still pending, but its inference slot is freed.
        await asyncio.sleep(0)  # let pipeline run to release_slot
        self.assertEqual(scheduler.inflight_count(), 0)
        self.assertEqual(scheduler.pending_count(), 1)
        self.assertTrue(await scheduler.spawn_if_slot(pipeline))
        self.assertEqual(scheduler.inflight_count(), 1)
        post_gate.set()
        await scheduler.drain()
        self.assertEqual(seen, [1, 2])


class TestSchedulerDrainAndCleanup(unittest.IsolatedAsyncioTestCase):
    def _build(self, **kwargs):
        cfg = TrajectorySchedulerConfig(max_on_fly=2, **kwargs)
        return cfg.build(sync_weights_interval=10)

    async def test_drain_waits_for_pending(self):
        scheduler = self._build()

        async def pipeline(req, release_slot):
            await asyncio.sleep(0.05)
            release_slot()

        for i in range(2):
            await scheduler.submit(_make_req(i))
        await scheduler.spawn_if_slot(pipeline)
        await scheduler.spawn_if_slot(pipeline)
        await scheduler.drain()
        self.assertEqual(scheduler.pending_count(), 0)

    async def test_pause_and_cleanup_cancels_on_timeout(self):
        scheduler = self._build(cleanup_timeout_s=0.2)
        cancelled = asyncio.Event()

        async def pipeline(req, release_slot):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            release_slot()

        await scheduler.submit(_make_req(1))
        await scheduler.spawn_if_slot(pipeline)
        elapsed = await scheduler.pause_and_cleanup()
        self.assertLess(elapsed, 2.0)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(scheduler.pending_count(), 0)
        self.assertEqual(scheduler.inflight_count(), 0)

    async def test_pause_and_cleanup_returns_quickly_if_empty(self):
        scheduler = self._build()
        elapsed = await scheduler.pause_and_cleanup()
        self.assertLess(elapsed, 0.1)


class TestClearQueueAndStaleness(unittest.IsolatedAsyncioTestCase):
    async def test_clear_queue_returns_in_order(self):
        cfg = TrajectorySchedulerConfig(max_on_fly=1)
        scheduler = cfg.build(sync_weights_interval=10)
        await scheduler.submit(_make_req(1))
        await scheduler.submit(_make_req(2))
        drained = await scheduler.clear_queue()
        self.assertEqual([r.prompt_uid for r in drained], [1, 2])
        self.assertEqual(scheduler.queue_len(), 0)

    async def test_is_model_expired_uses_threshold(self):
        cfg = TrajectorySchedulerConfig(max_on_fly=1, max_staleness=1)
        scheduler = cfg.build(sync_weights_interval=10)
        # threshold = (1+1) * 10 = 20
        self.assertFalse(scheduler.is_model_expired(train_step=0, model_step=0))
        self.assertTrue(scheduler.is_model_expired(train_step=100, model_step=0))

    async def test_pipeline_exception_removes_from_pending(self):
        cfg = TrajectorySchedulerConfig(max_on_fly=2)
        scheduler = cfg.build(sync_weights_interval=10)

        async def bad_pipeline(req, release_slot):
            raise RuntimeError("boom")

        await scheduler.submit(_make_req(1))
        await scheduler.spawn_if_slot(bad_pipeline)
        await scheduler.drain()
        # Wrapper's safety net releases the slot even when the pipeline raises.
        self.assertEqual(scheduler.pending_count(), 0)
        self.assertEqual(scheduler.inflight_count(), 0)


if __name__ == "__main__":
    unittest.main()
