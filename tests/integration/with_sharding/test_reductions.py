
import unittest
import numpy as np
import pytest
from nabla import Tensor, ops, reduce_sum, mean
from nabla.sharding import DeviceMesh, DimSpec, ShardingSpec
from nabla.core.compute_graph import GRAPH

class TestShardingReductions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        GRAPH._reset(None, 0)
        self.mesh = DeviceMesh("test_mesh", (4,), ("x",))

    async def test_reduce_sum_sharded_axis(self):
        """Test sum(axis=0) where axis 0 is sharded.
        
        Input: (M, N) sharded on M.
        Output: (N,) replicated.
        Requires AllReduce.
        """
        M, N = 16, 8
        axis = 0
        
        np_a = np.random.randn(M, N).astype(np.float32)
        a = Tensor.from_dlpack(np_a)
        a = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Sum over sharded dimension -> AllReduce
        c = reduce_sum(a, axis=axis)
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (N,))
        # Result should be replicated (no sharded dims left)
        self.assertTrue(all(not d.axes for d in c._impl.sharding.dim_specs))
        
        expected = np.sum(np_a, axis=axis)
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    async def test_reduce_sum_replicated_axis(self):
        """Test sum(axis=1) where axis 1 is replicated.
        
        Input: (M, N) sharded on M.
        Output: (M,) sharded on M.
        Requires local reduction only.
        """
        M, N = 16, 8
        axis = 1
        
        np_a = np.random.randn(M, N).astype(np.float32)
        a = Tensor.from_dlpack(np_a)
        a = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        c = reduce_sum(a, axis=axis)
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (M,))
        # Sharding on M should be preserved
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np.sum(np_a, axis=axis)
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    async def test_reduce_mean_keepdims_sharded(self):
        """Test mean(axis=0, keepdims=True) where axis 0 is sharded.
        
        Input: (M, N) sharded on M.
        Output: (1, N) replicated? Or partially sharded?
        """
        M, N = 16, 8
        axis = 0
        
        np_a = np.random.randn(M, N).astype(np.float32)
        a = Tensor.from_dlpack(np_a)
        a = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        c = mean(a, axis=axis, keepdims=True)
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (1, N))
        # dim 0 should be replicated (empty axes) because size 1
        self.assertFalse(c._impl.sharding.dim_specs[0].axes)
        
        expected = np.mean(np_a, axis=axis, keepdims=True)
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    async def test_reduce_sum_multi_axis(self):
        """Test sum over multiple axes.
        
        Input: (B, M, N) sharded on B.
        Reduce sum over (0, 2) -> (M,).
        """
        B, M, N = 4, 8, 8
        
        np_a = np.random.randn(B, M, N).astype(np.float32)
        a = Tensor.from_dlpack(np_a)
        a = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
        
        # Reduce N first (local) -> (B, M)
        temp = reduce_sum(a, axis=2)
        # Reduce B next (AllReduce) -> (M,)
        c = reduce_sum(temp, axis=0)
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (M,))
        self.assertTrue(all(not d.axes for d in c._impl.sharding.dim_specs))
        
        expected = np.sum(np_a, axis=(0, 2))
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

if __name__ == "__main__":
    unittest.main()
