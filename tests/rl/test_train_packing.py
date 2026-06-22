"""Unit tests for RL train-data packing and optimizer-step grouping.

These tests cover the logic that guarantees every dp rank performs exactly
``optimizer_steps`` optimizer updates:

- :func:`xtuner.v1.rl.trainer.worker.compute_group_bounds` splits the per-rank
  packs into exactly ``optimizer_steps`` groups.
- :meth:`xtuner.v1.rl.trainer.controller.TrainingController._get_balanced_pack_infos`
  distributes sequences into the required number of packs without exceeding the
  ``pack_max_length`` capacity.

No rollout engine, Ray cluster, or GPU is required.
"""

import unittest

from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.worker import compute_group_bounds


class TestComputeGroupBounds(unittest.TestCase):
    def test_exact_number_of_groups(self):
        # The number of optimizer updates must always equal `num_groups`, regardless of whether
        # `num_items` is a clean multiple of it. This is the core bug the change fixes.
        for num_items in [8, 10, 16, 17, 64]:
            bounds = compute_group_bounds(num_items, 8)
            self.assertEqual(len(bounds), 8)

    def test_groups_cover_all_items_contiguously(self):
        bounds = compute_group_bounds(17, 8)
        # Contiguous, non-overlapping, covering [0, num_items).
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 17)
        for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:]):
            self.assertEqual(prev_end, next_start)

    def test_groups_are_balanced_and_non_empty(self):
        bounds = compute_group_bounds(17, 8)
        sizes = [end - start for start, end in bounds]
        self.assertEqual(sum(sizes), 17)
        # Sizes differ by at most one and none is empty (since num_items >= num_groups).
        self.assertEqual(max(sizes) - min(sizes), 1)
        self.assertGreaterEqual(min(sizes), 1)

    def test_equal_when_multiple(self):
        bounds = compute_group_bounds(16, 8)
        sizes = [end - start for start, end in bounds]
        self.assertEqual(sizes, [2] * 8)


class TestBalancedPackInfos(unittest.TestCase):
    def setUp(self):
        # `_get_balanced_pack_infos` does not touch `self.workers` / `self.logger`,
        # so an empty controller is sufficient.
        self.controller = TrainingController(workers=[])

    def _all_indices(self, pack_infos):
        flat = [i for info in pack_infos for i in info["indices"]]
        return sorted(flat)

    def test_every_sequence_assigned_exactly_once(self):
        num_tokens = [100, 200, 300, 400, 500, 600, 700, 800]
        pack_infos = self.controller._get_balanced_pack_infos(
            num_tokens, pack_max_length=4096, num_bins=4, dp_size=2
        )
        self.assertEqual(self._all_indices(pack_infos), list(range(len(num_tokens))))

    def test_capacity_never_exceeded(self):
        num_tokens = [500] * 20
        pack_max_length = 1000
        pack_infos = self.controller._get_balanced_pack_infos(
            num_tokens, pack_max_length=pack_max_length, num_bins=4, dp_size=2
        )
        for info in pack_infos:
            total = sum(num_tokens[i] for i in info["indices"])
            self.assertLessEqual(total, pack_max_length)

    def test_num_bins_respected_when_capacity_allows(self):
        # 8 short sequences fit easily into 8 bins -> no growth needed.
        num_tokens = [100] * 8
        pack_infos = self.controller._get_balanced_pack_infos(
            num_tokens, pack_max_length=4096, num_bins=8, dp_size=4
        )
        self.assertEqual(len(pack_infos), 8)

    def test_num_bins_grows_by_dp_size_when_capacity_insufficient(self):
        # Total tokens = 20 * 500 = 10000; with pack_max_length=1000 each bin holds <= 2 seqs,
        # so at least 10 bins are required. Starting from 4 bins, it must grow in steps of dp_size=2.
        num_tokens = [500] * 20
        dp_size = 2
        pack_infos = self.controller._get_balanced_pack_infos(
            num_tokens, pack_max_length=1000, num_bins=4, dp_size=dp_size
        )
        self.assertGreaterEqual(len(pack_infos), 10)
        self.assertEqual(len(pack_infos) % dp_size, 0)
        self.assertEqual(self._all_indices(pack_infos), list(range(len(num_tokens))))

    def test_balanced_load(self):
        # LPT keeps the spread between the most- and least-loaded packs within one max item.
        num_tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        num_bins = 5
        pack_infos = self.controller._get_balanced_pack_infos(
            num_tokens, pack_max_length=10000, num_bins=num_bins, dp_size=1
        )
        loads = [sum(num_tokens[i] for i in info["indices"]) for info in pack_infos]
        self.assertLessEqual(max(loads) - min(loads), max(num_tokens))

    def test_empty_bins_when_fewer_sequences_than_bins(self):
        # Degenerate case: fewer sequences than required packs -> some packs are empty.
        num_tokens = [100, 200, 300]
        num_bins = 8
        pack_infos = self.controller._get_balanced_pack_infos(
            num_tokens, pack_max_length=4096, num_bins=num_bins, dp_size=4
        )
        self.assertEqual(len(pack_infos), num_bins)
        empty = [info for info in pack_infos if len(info["indices"]) == 0]
        self.assertEqual(len(empty), num_bins - len(num_tokens))
        self.assertEqual(self._all_indices(pack_infos), list(range(len(num_tokens))))


if __name__ == "__main__":
    unittest.main()
