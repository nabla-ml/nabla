
import unittest
import numpy as np
import pytest
from nabla import Tensor, reshape, swap_axes, unsqueeze, squeeze
from nabla.sharding import DeviceMesh, DimSpec, ShardingSpec
from nabla.core.compute_graph import GRAPH

class TestShardingViews(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        GRAPH._reset(None, 0)
        self.mesh = DeviceMesh("test_mesh", (4,), ("x",))

    async def test_reshape_split_sharded_dim(self):
        """Test (16,) sharded on 0 -> (4, 4) sharded on 0.
        
        Splitting a sharded dimension into two, keeping sharding on the outer one.
        Input: (16,) sharded on "x". Local (4,).
        Output: (4, 4) sharded on "x". Local (1, 4).
        
        Logic:
        16 elements. Shards: [0-3], [4-7], [8-11], [12-15].
        Reshape (4, 4):
        Row 0: 0-3. (Shard 0)
        Row 1: 4-7. (Shard 1)
        ...
        So Shard 0 holds Row 0. Local (1, 4).
        This is valid local reshape `(4,) -> (1, 4)`.
        """
        N = 16
        
        np_a = np.arange(N).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"])])
        
        c = reshape(a, (4, 4))
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (4, 4))
        # Expect sharding on dim 0
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        self.assertFalse(c._impl.sharding.dim_specs[1].axes)
        
        expected = np_a.reshape(4, 4)
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected)

    async def test_reshape_merge_sharded_dim(self):
        """Test (4, 4) sharded on 0 -> (16,) sharded on 0.
        
        Merging a sharded dimension.
        Input: (4, 4) sharded on "x". (1, 4) per shard.
        Output: (16,) sharded on "x". (4,) per shard.
        
        Valid local reshape.
        """
        np_a = np.arange(16).astype(np.float32).reshape(4, 4)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        c = reshape(a, (16,))
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (16,))
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a.reshape(16)
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected)

    async def test_transpose_swap_sharding(self):
        """Test (4, 8) sharded on 0 -> (8, 4) sharded on 1.
        
        Input: (4, 8) sharded on "x". Local (1, 8).
        Swap 0, 1.
        Output: (8, 4) sharded on 1. Local (8, 1).
        
        Valid local transpose.
        """
        np_a = np.random.randn(4, 8).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        c = swap_axes(a, 0, 1)
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (8, 4))
        # Sharding should move to dim 1
        self.assertFalse(c._impl.sharding.dim_specs[0].axes)
        self.assertEqual(c._impl.sharding.dim_specs[1].axes, ["x"])
        
        expected = np.transpose(np_a, (1, 0))
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected)

    async def test_squeeze_unsqueeze_sharded(self):
        """Test unsqueeze (new dim) and squeeze (remove dim) with sharding."""
        # 1. Unsqueeze: (16,) sharded -> (1, 16) sharded on dim 1
        np_a = np.arange(16).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"])])
        
        b = unsqueeze(a, axis=0)
        c = squeeze(b, axis=0)
        
        await c.realize
        
        # Check b (intermediate) properties via inspection if possible, or just trust c
        # Since b is not realized, we can't easily check its shape/sharding if it relies on propagation?
        # Propagation happens eagerly! So b._impl.sharding should be correct immediately.
        
        self.assertEqual(tuple(int(d) for d in b.shape), (1, 16))
        # New dim 0 should be replicated (empty axes)
        self.assertFalse(b._impl.sharding.dim_specs[0].axes)
        # Dim 1 should preserve sharding
        self.assertEqual(b._impl.sharding.dim_specs[1].axes, ["x"])
        
        self.assertEqual(tuple(int(d) for d in c.shape), (16,))
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected)

if __name__ == "__main__":
    unittest.main()
