"""Unit tests for :meth:`ReplayBuffer.take_batch_by_trajectory_count`.

Uses the same ``MockState`` duck-typing pattern as
``tests/rl/test_replay_buffer.py`` so new tests are consistent with the
existing buffer test suite.
"""

import unittest

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig


class _MockState:
    def __init__(self, uid: int, status: Status = Status.COMPLETED, staleness: int = 0):
        self.uid = uid
        self.status = status
        self.seq_staleness = staleness
        self.response_ids: list[int] = []
        self.response_model_steps: list[int] | None = None


class TestTakeBatchByTrajectoryCount(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_returns_oldest_first(self):
        rb = SyncReplayBufferConfig().build()
        for i in range(3):
            base = i * 10
            await rb.put([_MockState(base), _MockState(base + 1)], "t")

        batch, group_counts, traj_counts = await rb.take_batch_by_trajectory_count({"t": 3})
        self.assertEqual(group_counts["t"], 2)
        self.assertEqual(traj_counts["t"], 4)
        uids = sorted(s.uid for group in batch["t"] for s in group)
        self.assertEqual(uids, [0, 1, 10, 11])

    async def test_zero_target_returns_empty(self):
        rb = SyncReplayBufferConfig().build()
        await rb.put([_MockState(1)], "t")
        batch, group_counts, traj_counts = await rb.take_batch_by_trajectory_count({"t": 0})
        self.assertEqual(batch["t"], [])
        self.assertEqual(group_counts["t"], 0)
        self.assertEqual(traj_counts["t"], 0)

    async def test_target_larger_than_available_drains_all(self):
        rb = SyncReplayBufferConfig().build()
        for i in range(2):
            await rb.put([_MockState(i)], "t")
        batch, group_counts, traj_counts = await rb.take_batch_by_trajectory_count({"t": 100})
        self.assertEqual(group_counts["t"], 2)
        self.assertEqual(traj_counts["t"], 2)

    async def test_last_group_fully_included_overflows(self):
        rb = SyncReplayBufferConfig().build()
        await rb.put([_MockState(i) for i in range(4)], "t")
        await rb.put([_MockState(i + 10) for i in range(4)], "t")
        batch, group_counts, traj_counts = await rb.take_batch_by_trajectory_count({"t": 5})
        # First group of 4 < 5; pull second group entirely, total = 8.
        self.assertEqual(group_counts["t"], 2)
        self.assertEqual(traj_counts["t"], 8)

    async def test_task_isolation(self):
        rb = SyncReplayBufferConfig().build()
        await rb.put([_MockState(1)], "t1")
        await rb.put([_MockState(2)], "t2")
        batch, _, _ = await rb.take_batch_by_trajectory_count({"t1": 1, "t2": 1})
        self.assertEqual(len(batch["t1"]), 1)
        self.assertEqual(len(batch["t2"]), 1)
        self.assertEqual(batch["t1"][0][0].uid, 1)
        self.assertEqual(batch["t2"][0][0].uid, 2)

    async def test_staleness_policy_prefers_staler_group(self):
        rb = AsyncReplayBufferConfig().build()
        fresher = _MockState(2, staleness=1)
        older = _MockState(1, staleness=5)
        await rb.put([fresher], "t")
        await rb.put([older], "t")
        batch, _, _ = await rb.take_batch_by_trajectory_count({"t": 1})
        self.assertEqual(batch["t"][0][0].uid, 1)

    async def test_status_filter(self):
        rb = SyncReplayBufferConfig().build()
        await rb.put([_MockState(1, status=Status.FAILED)], "t")
        await rb.put([_MockState(2, status=Status.COMPLETED)], "t")
        batch, group_counts, _ = await rb.take_batch_by_trajectory_count(
            {"t": 1}, group_status=Status.COMPLETED
        )
        self.assertEqual(group_counts["t"], 1)
        self.assertEqual(batch["t"][0][0].uid, 2)

    async def test_selected_groups_are_removed_from_storage(self):
        rb = SyncReplayBufferConfig().build()
        for i in range(2):
            await rb.put([_MockState(i)], "t")
        await rb.take_batch_by_trajectory_count({"t": 1})
        remaining = await rb.count(task_name="t", group_status=Status.COMPLETED)
        self.assertEqual(remaining, 1)


if __name__ == "__main__":
    unittest.main()
