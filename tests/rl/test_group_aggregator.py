"""Unit tests for :mod:`xtuner.v1.rl.agent_loop_manager.group_aggregator`.

Uses minimal :class:`RolloutState` instances (empty ``message`` list) to
exercise the aggregator; no rollout engine or sampler is required.
"""

import unittest

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop_manager.group_aggregator import GroupAggregator
from xtuner.v1.rl.agent_loop_manager.group_policy import GroupPolicyConfig, GroupState


def _make_prompt(message_uid: int) -> RolloutState:
    return RolloutState(message=[], message_uid=message_uid)


def _make_traj(message_uid: int, uid: int, score: float) -> RolloutState:
    return RolloutState(
        message=[],
        message_uid=message_uid,
        uid=uid,
        reward={"score": score},
        status=Status.COMPLETED,
    )


def _make_aborted(message_uid: int, uid: int) -> RolloutState:
    return RolloutState(
        message=[],
        message_uid=message_uid,
        uid=uid,
        status=Status.ABORTED,
    )


class TestGroupAggregatorRegistration(unittest.IsolatedAsyncioTestCase):
    def _build(self, min_repeat=2, max_repeat=8) -> GroupAggregator:
        policy = GroupPolicyConfig(min_repeat=min_repeat, max_repeat=max_repeat).build()
        return GroupAggregator(policy)

    async def test_register_creates_aggregation(self):
        agg = self._build()
        result = await agg.register_prompt(_make_prompt(42), "gsm8k")
        self.assertEqual(result.prompt_uid, 42)
        self.assertEqual(result.task_name, "gsm8k")
        self.assertEqual(result.min_repeat, 2)
        self.assertEqual(result.max_repeat, 8)
        self.assertEqual(await agg.active_count(), 1)

    async def test_duplicate_registration_raises(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        with self.assertRaises(KeyError):
            await agg.register_prompt(_make_prompt(1), "task")

    async def test_missing_message_uid_raises(self):
        agg = self._build()
        prompt = RolloutState(message=[])
        with self.assertRaises(ValueError):
            await agg.register_prompt(prompt, "task")


class TestAddTrajectory(unittest.IsolatedAsyncioTestCase):
    def _build(self, min_repeat=2, max_repeat=8) -> GroupAggregator:
        policy = GroupPolicyConfig(min_repeat=min_repeat, max_repeat=max_repeat).build()
        return GroupAggregator(policy)

    async def test_collecting(self):
        agg = self._build(min_repeat=4, max_repeat=8)
        await agg.register_prompt(_make_prompt(1), "task")
        state, group = await agg.add_trajectory(_make_traj(1, 10, 1.0))
        self.assertIs(state, GroupState.COLLECTING)
        self.assertIsNone(group)

    async def test_ready_finalizes_and_removes_aggregation(self):
        agg = self._build(min_repeat=2, max_repeat=8)
        await agg.register_prompt(_make_prompt(1), "task")
        state1, _ = await agg.add_trajectory(_make_traj(1, 10, 1.0))
        state2, group = await agg.add_trajectory(_make_traj(1, 11, 0.0))
        self.assertIs(state1, GroupState.COLLECTING)
        self.assertIs(state2, GroupState.READY)
        self.assertIsNotNone(group)
        self.assertEqual([t.uid for t in group], [10, 11])
        self.assertFalse(await agg.exists(1))

    async def test_needs_more_when_all_equal(self):
        agg = self._build(min_repeat=2, max_repeat=8)
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.add_trajectory(_make_traj(1, 10, 0.0))
        state, group = await agg.add_trajectory(_make_traj(1, 11, 0.0))
        self.assertIs(state, GroupState.NEEDS_MORE)
        self.assertIsNone(group)
        self.assertTrue(await agg.exists(1))

    async def test_stopped_at_max_all_equal(self):
        agg = self._build(min_repeat=2, max_repeat=2)
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.add_trajectory(_make_traj(1, 10, 0.5))
        state, group = await agg.add_trajectory(_make_traj(1, 11, 0.5))
        self.assertIs(state, GroupState.STOPPED)
        self.assertIsNone(group)

    async def test_late_trajectory_orphaned(self):
        agg = self._build(min_repeat=2, max_repeat=8)
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.add_trajectory(_make_traj(1, 10, 1.0))
        await agg.add_trajectory(_make_traj(1, 11, 0.0))  # READY, removes agg
        state, group = await agg.add_trajectory(_make_traj(1, 12, 1.0))
        self.assertIsNone(state)
        self.assertIsNone(group)

    async def test_missing_message_uid_raises(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        traj = RolloutState(message=[], uid=10, reward={"score": 1.0})
        with self.assertRaises(ValueError):
            await agg.add_trajectory(traj)


class TestInFlightBookkeeping(unittest.IsolatedAsyncioTestCase):
    def _build(self) -> GroupAggregator:
        policy = GroupPolicyConfig(min_repeat=2, max_repeat=8).build()
        return GroupAggregator(policy)

    async def test_positive_and_negative_deltas(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.mark_in_flight(1, +1)
        await agg.mark_in_flight(1, +2)
        snap = await agg.get_snapshot(1)
        assert snap is not None
        self.assertEqual(snap.in_flight, 3)
        await agg.mark_in_flight(1, -1)
        snap = await agg.get_snapshot(1)
        assert snap is not None
        self.assertEqual(snap.in_flight, 2)

    async def test_negative_goes_below_zero_raises(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.mark_in_flight(1, +1)
        with self.assertRaises(RuntimeError):
            await agg.mark_in_flight(1, -2)

    async def test_missing_aggregation_noop(self):
        agg = self._build()
        await agg.mark_in_flight(42, -1)
        self.assertEqual(await agg.active_count(), 0)


class TestPendingKeep(unittest.IsolatedAsyncioTestCase):
    def _build(self) -> GroupAggregator:
        policy = GroupPolicyConfig(min_repeat=2, max_repeat=8).build()
        return GroupAggregator(policy)

    async def test_push_pop(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        aborted = _make_aborted(1, 50)
        await agg.push_pending_keep(aborted)
        self.assertTrue(await agg.has_pending_keep(1))
        popped = await agg.pop_pending_keep(1)
        assert popped is not None
        self.assertEqual(popped.uid, 50)
        self.assertFalse(await agg.has_pending_keep(1))

    async def test_fifo_order(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.push_pending_keep(_make_aborted(1, 100))
        await agg.push_pending_keep(_make_aborted(1, 101))
        first = await agg.pop_pending_keep(1)
        second = await agg.pop_pending_keep(1)
        assert first is not None and second is not None
        self.assertEqual(first.uid, 100)
        self.assertEqual(second.uid, 101)

    async def test_orphan_push_silently_discarded(self):
        agg = self._build()
        await agg.push_pending_keep(_make_aborted(99, 50))
        self.assertFalse(await agg.has_pending_keep(99))

    async def test_pop_empty_returns_none(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        self.assertIsNone(await agg.pop_pending_keep(1))


class TestDropAndSnapshot(unittest.IsolatedAsyncioTestCase):
    def _build(self) -> GroupAggregator:
        policy = GroupPolicyConfig(min_repeat=2, max_repeat=8).build()
        return GroupAggregator(policy)

    async def test_drop_removes(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.drop(1)
        self.assertFalse(await agg.exists(1))

    async def test_drop_unknown_noop(self):
        agg = self._build()
        await agg.drop(42)

    async def test_snapshot_is_shallow_copy(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.add_trajectory(_make_traj(1, 10, 1.0))
        snap = await agg.get_snapshot(1)
        assert snap is not None
        snap.completed.append(_make_traj(1, 99, 0.0))
        # Mutation to snapshot does not leak back
        snap2 = await agg.get_snapshot(1)
        assert snap2 is not None
        self.assertEqual(len(snap2.completed), 1)


class TestStateDictRoundTrip(unittest.IsolatedAsyncioTestCase):
    def _build(self) -> GroupAggregator:
        policy = GroupPolicyConfig(min_repeat=2, max_repeat=8).build()
        return GroupAggregator(policy)

    async def test_roundtrip_preserves_completed_and_pending(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.add_trajectory(_make_traj(1, 10, 0.0))
        await agg.push_pending_keep(_make_aborted(1, 11))
        await agg.mark_in_flight(1, +2)

        state = await agg.state_dict()

        fresh = self._build()
        await fresh.load_state_dict(state)
        snap = await fresh.get_snapshot(1)
        assert snap is not None
        self.assertEqual(len(snap.completed), 1)
        self.assertEqual(snap.completed[0].uid, 10)
        self.assertEqual(len(snap.pending_keep), 1)
        self.assertEqual(snap.pending_keep[0].uid, 11)
        # in_flight is reset on resume
        self.assertEqual(snap.in_flight, 0)

    async def test_load_replaces_existing(self):
        agg = self._build()
        await agg.register_prompt(_make_prompt(1), "task")
        await agg.load_state_dict({"groups": []})
        self.assertEqual(await agg.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
