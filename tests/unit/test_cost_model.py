"""Unit tests for communication cost model."""

import unittest

from nabla.sharding.spec import DeviceMesh
from nabla.sharding.cost_model import (
    allreduce_cost,
    allgather_cost,
    reduce_scatter_cost,
    resharding_cost,
)


class TestCommunicationCostModel(unittest.TestCase):
    """Tests for collective operation cost functions."""

    def setUp(self):
        # 4-device mesh for testing
        self.mesh = DeviceMesh("test", (4,), ("d",), devices=[0, 1, 2, 3])
        # 2D mesh: 2x4 = 8 devices
        self.mesh_2d = DeviceMesh("test2d", (2, 4), ("dp", "tp"), 
                                   devices=list(range(8)))

    def test_allreduce_cost_scales_with_size(self):
        """Larger tensors should have higher AllReduce cost."""
        cost_small = allreduce_cost(1000, self.mesh, ["d"])
        cost_large = allreduce_cost(10000, self.mesh, ["d"])
        
        # Cost should scale linearly with size
        self.assertGreater(cost_large, cost_small)
        self.assertAlmostEqual(cost_large / cost_small, 10.0, places=5)
        
    def test_allreduce_cost_scales_with_devices(self):
        """More devices should have higher AllReduce cost (more communication)."""
        # Create meshes with different device counts
        mesh_2 = DeviceMesh("m2", (2,), ("d",), devices=[0, 1])
        mesh_8 = DeviceMesh("m8", (8,), ("d",), devices=list(range(8)))
        
        cost_2 = allreduce_cost(1000, mesh_2, ["d"])
        cost_8 = allreduce_cost(1000, mesh_8, ["d"])
        
        # Cost formula: 2 * (n-1)/n * size
        # n=2: 2 * 0.5 * 1000 = 1000
        # n=8: 2 * 0.875 * 1000 = 1750
        self.assertGreater(cost_8, cost_2)
        
    def test_allreduce_cost_zero_for_single_device(self):
        """AllReduce on single device should be zero cost."""
        mesh_1 = DeviceMesh("m1", (1,), ("d",), devices=[0])
        cost = allreduce_cost(1000, mesh_1, ["d"])
        self.assertEqual(cost, 0.0)
        
    def test_allreduce_cost_zero_for_empty_axes(self):
        """AllReduce with no axes should be zero cost."""
        cost = allreduce_cost(1000, self.mesh, [])
        self.assertEqual(cost, 0.0)

    def test_allgather_cost_formula(self):
        """AllGather cost should follow (n-1)/n * total_size formula."""
        size = 1000
        cost = allgather_cost(size, self.mesh, ["d"])
        
        # Expected: (4-1)/4 * 1000 * 4 = 0.75 * 4000 = 3000
        expected = (4 - 1) / 4 * size * 4
        self.assertAlmostEqual(cost, expected, places=5)

    def test_reduce_scatter_cost_formula(self):
        """ReduceScatter cost should follow (n-1)/n * size formula."""
        size = 1000
        cost = reduce_scatter_cost(size, self.mesh, ["d"])
        
        # Expected: (4-1)/4 * 1000 = 750
        expected = (4 - 1) / 4 * size
        self.assertAlmostEqual(cost, expected, places=5)

    def test_resharding_cost_zero_if_both_none(self):
        """Resharding between None specs should be zero."""
        cost = resharding_cost(None, None, 1000, self.mesh)
        self.assertEqual(cost, 0.0)

    def test_resharding_cost_zero_for_shard_only(self):
        """Resharding from unsharded to sharded is free (just local slicing)."""
        from nabla.sharding.spec import DimSpec, ShardingSpec
        
        to_spec = ShardingSpec(self.mesh, [DimSpec(["d"]), DimSpec([])])
        cost = resharding_cost(None, to_spec, 1000, self.mesh)
        self.assertEqual(cost, 0.0)

    def test_resharding_cost_nonzero_for_gather(self):
        """Resharding from sharded to unsharded requires AllGather."""
        from nabla.sharding.spec import DimSpec, ShardingSpec
        
        from_spec = ShardingSpec(self.mesh, [DimSpec(["d"]), DimSpec([])])
        cost = resharding_cost(from_spec, None, 4000, self.mesh)  # 4000 bytes global
        
        # Should be positive (AllGather cost)
        self.assertGreater(cost, 0.0)

    def test_bandwidth_affects_cost(self):
        """Higher bandwidth should result in lower cost."""
        mesh_slow = DeviceMesh("slow", (4,), ("d",), devices=[0, 1, 2, 3], bandwidth=1.0)
        mesh_fast = DeviceMesh("fast", (4,), ("d",), devices=[0, 1, 2, 3], bandwidth=2.0)
        
        cost_slow = allreduce_cost(1000, mesh_slow, ["d"])
        cost_fast = allreduce_cost(1000, mesh_fast, ["d"])
        
        # Faster bandwidth = lower cost
        self.assertLess(cost_fast, cost_slow)
        self.assertAlmostEqual(cost_slow / cost_fast, 2.0, places=5)

    def test_2d_mesh_multi_axis(self):
        """Cost should increase when reducing over multiple axes."""
        # Reduce over just "dp" (2 devices)
        cost_dp = allreduce_cost(1000, self.mesh_2d, ["dp"])
        
        # Reduce over just "tp" (4 devices)  
        cost_tp = allreduce_cost(1000, self.mesh_2d, ["tp"])
        
        # Reduce over both (8 devices)
        cost_both = allreduce_cost(1000, self.mesh_2d, ["dp", "tp"])
        
        # More devices = higher communication cost
        self.assertLess(cost_dp, cost_both)
        self.assertLess(cost_tp, cost_both)


if __name__ == "__main__":
    unittest.main()
