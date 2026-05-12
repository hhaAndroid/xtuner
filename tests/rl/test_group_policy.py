"""Unit tests for :mod:`xtuner.v1.rl.agent_loop_manager.group_policy`.

Pure judgment logic; tests use lightweight mocks rather than full
``RolloutState`` objects.
"""

import unittest
from dataclasses import dataclass, field

from xtuner.v1.rl.agent_loop_manager.group_policy import (
    DefaultGroupPolicy,
    GroupPolicyConfig,
    GroupState,
)


@dataclass
class _MockAgg:
    completed: list = field(default_factory=list)
    in_flight: int = 0


@dataclass
class _MockTraj:
    reward: dict | None
    uid: int = 0


class TestGroupPolicyConfig(unittest.TestCase):
    def test_valid_config(self):
        cfg = GroupPolicyConfig(min_repeat=2, max_repeat=8)
        self.assertEqual(cfg.min_repeat, 2)
        self.assertEqual(cfg.max_repeat, 8)
        self.assertEqual(cfg.score_key, "score")

    def test_min_repeat_below_two_rejected(self):
        with self.assertRaises(Exception):
            GroupPolicyConfig(min_repeat=1, max_repeat=8)

    def test_max_smaller_than_min_rejected(self):
        with self.assertRaises(Exception):
            GroupPolicyConfig(min_repeat=4, max_repeat=2)

    def test_build_returns_default_policy(self):
        cfg = GroupPolicyConfig(min_repeat=2, max_repeat=4)
        policy = cfg.build()
        self.assertIsInstance(policy, DefaultGroupPolicy)
        self.assertEqual(policy.min_repeat, 2)
        self.assertEqual(policy.max_repeat, 4)


class TestDefaultGroupPolicyDecisions(unittest.TestCase):
    def _build(self, min_repeat=2, max_repeat=8, tol=1e-8):
        return GroupPolicyConfig(
            min_repeat=min_repeat, max_repeat=max_repeat, score_tol=tol
        ).build()

    def test_collecting_below_min_repeat(self):
        policy = self._build(min_repeat=4, max_repeat=8)
        agg = _MockAgg(completed=[_MockTraj({"score": 1.0})], in_flight=0)
        self.assertIs(policy.on_trajectory_done(agg), GroupState.COLLECTING)

    def test_ready_when_scores_differ(self):
        policy = self._build(min_repeat=2, max_repeat=8)
        agg = _MockAgg(completed=[
            _MockTraj({"score": 1.0}),
            _MockTraj({"score": 0.0}),
        ])
        self.assertIs(policy.on_trajectory_done(agg), GroupState.READY)

    def test_needs_more_all_equal_with_headroom(self):
        policy = self._build(min_repeat=2, max_repeat=8)
        agg = _MockAgg(completed=[
            _MockTraj({"score": 1.0}),
            _MockTraj({"score": 1.0}),
        ], in_flight=0)
        self.assertIs(policy.on_trajectory_done(agg), GroupState.NEEDS_MORE)

    def test_stopped_at_max_repeat_all_equal(self):
        policy = self._build(min_repeat=2, max_repeat=2)
        agg = _MockAgg(completed=[
            _MockTraj({"score": 0.0}),
            _MockTraj({"score": 0.0}),
        ], in_flight=0)
        self.assertIs(policy.on_trajectory_done(agg), GroupState.STOPPED)

    def test_stopped_counts_in_flight_toward_max(self):
        policy = self._build(min_repeat=2, max_repeat=4)
        agg = _MockAgg(completed=[
            _MockTraj({"score": 0.5}),
            _MockTraj({"score": 0.5}),
        ], in_flight=2)
        self.assertIs(policy.on_trajectory_done(agg), GroupState.STOPPED)

    def test_tolerance_absorbs_small_differences(self):
        policy = self._build(min_repeat=2, max_repeat=8, tol=0.01)
        agg = _MockAgg(completed=[
            _MockTraj({"score": 1.0}),
            _MockTraj({"score": 1.005}),
        ])
        self.assertIs(policy.on_trajectory_done(agg), GroupState.NEEDS_MORE)

    def test_missing_reward_raises(self):
        policy = self._build()
        agg = _MockAgg(completed=[_MockTraj(None), _MockTraj(None)])
        with self.assertRaises(RuntimeError):
            policy.on_trajectory_done(agg)

    def test_missing_score_key_raises(self):
        policy = self._build()
        agg = _MockAgg(completed=[
            _MockTraj({"acc": 1.0}),
            _MockTraj({"acc": 0.0}),
        ])
        with self.assertRaises(KeyError):
            policy.on_trajectory_done(agg)


class TestShouldSpawnMore(unittest.TestCase):
    def _build(self, min_repeat=2, max_repeat=8):
        return GroupPolicyConfig(min_repeat=min_repeat, max_repeat=max_repeat).build()

    def test_below_min_requests_gap(self):
        policy = self._build(min_repeat=4, max_repeat=8)
        agg = _MockAgg(completed=[], in_flight=1)
        self.assertEqual(policy.should_spawn_more(agg), 3)

    def test_at_min_requests_headroom_to_max(self):
        policy = self._build(min_repeat=4, max_repeat=8)
        agg = _MockAgg(completed=[_MockTraj({"score": 1})] * 4, in_flight=0)
        self.assertEqual(policy.should_spawn_more(agg), 4)

    def test_at_max_returns_zero(self):
        policy = self._build(min_repeat=2, max_repeat=4)
        agg = _MockAgg(completed=[_MockTraj({"score": 1})] * 4, in_flight=0)
        self.assertEqual(policy.should_spawn_more(agg), 0)


if __name__ == "__main__":
    unittest.main()
