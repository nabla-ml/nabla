# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Unit tests for communication cost model."""

import unittest

from nabla.core.sharding.spec import DeviceMesh
from nabla.ops.communication import (
    AllGatherOp,
    AllReduceOp,
    ReduceScatterOp,
    ReshardOp,
)


class TestCommunicationCostModel(unittest.TestCase):
    """Tests for collective operation cost functions."""

    def setUp(self):
        self.mesh = DeviceMesh("test", (4,), ("d",), devices=[0, 1, 2, 3])

        self.mesh_2d = DeviceMesh(
            "test2d", (2, 4), ("dp", "tp"), devices=list(range(8))
        )

    def test_allreduce_cost_scales_with_size(self):
        """Larger tensors should have higher AllReduce cost."""
        cost_small = AllReduceOp.estimate_cost(1000, self.mesh, ["d"])
        cost_large = AllReduceOp.estimate_cost(10000, self.mesh, ["d"])

        self.assertGreater(cost_large, cost_small)
        self.assertAlmostEqual(cost_large / cost_small, 10.0, places=5)

    def test_allreduce_cost_scales_with_devices(self):
        """More devices should have higher AllReduce cost (more communication)."""

        mesh_2 = DeviceMesh("m2", (2,), ("d",), devices=[0, 1])
        mesh_8 = DeviceMesh("m8", (8,), ("d",), devices=list(range(8)))

        cost_2 = AllReduceOp.estimate_cost(1000, mesh_2, ["d"])
        cost_8 = AllReduceOp.estimate_cost(1000, mesh_8, ["d"])

        self.assertGreater(cost_8, cost_2)

    def test_allreduce_cost_zero_for_single_device(self):
        """AllReduce on single device should be zero cost."""
        mesh_1 = DeviceMesh("m1", (1,), ("d",), devices=[0])
        cost = AllReduceOp.estimate_cost(1000, mesh_1, ["d"])
        self.assertEqual(cost, 0.0)

    def test_allreduce_cost_zero_for_empty_axes(self):
        """AllReduce with no axes should be zero cost."""
        cost = AllReduceOp.estimate_cost(1000, self.mesh, [])
        self.assertEqual(cost, 0.0)

    def test_allgather_cost_formula(self):
        """AllGather cost should follow (n-1)/n * total_size formula."""
        size = 1000
        cost = AllGatherOp.estimate_cost(size, self.mesh, ["d"])

        expected = (4 - 1) / 4 * size
        self.assertAlmostEqual(cost, expected, places=5)

    def test_reduce_scatter_cost_formula(self):
        """ReduceScatter cost should follow (n-1)/n * size formula."""
        size = 1000
        cost = ReduceScatterOp.estimate_cost(size, self.mesh, ["d"])

        expected = (4 - 1) / 4 * size
        self.assertAlmostEqual(cost, expected, places=5)

    def test_resharding_cost_zero_if_both_none(self):
        """Resharding between None specs should be zero."""
        op = ReshardOp()

        cost = op.communication_cost([], [], [(1000,)], [(1000,)], self.mesh)
        self.assertEqual(cost, 0.0)

    def test_resharding_cost_zero_for_shard_only(self):
        """Resharding from unsharded to sharded is free (just local slicing)."""
        from nabla.core.sharding.spec import DimSpec, ShardingSpec

        to_spec = ShardingSpec(self.mesh, [DimSpec(["d"]), DimSpec([])])
        op = ReshardOp()

        cost = op.communication_cost([], [to_spec], [(1000,)], [(1000,)], self.mesh)
        self.assertEqual(cost, 0.0)

    def test_resharding_cost_nonzero_for_gather(self):
        """Resharding from sharded to unsharded requires AllGather."""
        from nabla.core.sharding.spec import DimSpec, ShardingSpec

        from_spec = ShardingSpec(self.mesh, [DimSpec(["d"]), DimSpec([])])
        op = ReshardOp()

        cost = op.communication_cost([from_spec], [], [(1000,)], [(1000,)], self.mesh)

        self.assertGreater(cost, 0.0)

    def test_bandwidth_affects_cost(self):
        """Higher bandwidth should result in lower cost."""
        mesh_slow = DeviceMesh(
            "slow", (4,), ("d",), devices=[0, 1, 2, 3], bandwidth=1.0
        )
        mesh_fast = DeviceMesh(
            "fast", (4,), ("d",), devices=[0, 1, 2, 3], bandwidth=2.0
        )

        cost_slow = AllReduceOp.estimate_cost(1000, mesh_slow, ["d"])
        cost_fast = AllReduceOp.estimate_cost(1000, mesh_fast, ["d"])

        self.assertLess(cost_fast, cost_slow)
        self.assertAlmostEqual(cost_slow / cost_fast, 2.0, places=5)

    def test_2d_mesh_multi_axis(self):
        """Cost should increase when reducing over multiple axes."""

        cost_dp = AllReduceOp.estimate_cost(1000, self.mesh_2d, ["dp"])

        cost_tp = AllReduceOp.estimate_cost(1000, self.mesh_2d, ["tp"])

        cost_both = AllReduceOp.estimate_cost(1000, self.mesh_2d, ["dp", "tp"])

        self.assertLess(cost_dp, cost_both)
        self.assertLess(cost_tp, cost_both)


if __name__ == "__main__":
    unittest.main()
