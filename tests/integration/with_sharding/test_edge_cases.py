"""Edge Case Tests for Sharding Robustness.

Tests edge cases that could break distributed computing:
- Uneven tensor dimensions (prime numbers not divisible by shard count)
- Very small tensors (2x2, 3x3)
- Single-element tensors
- Complex mesh scenarios (asymmetric, 3D)
- Communication ops on edge cases
"""

import unittest
import numpy as np
import pytest
from nabla import Tensor
from nabla.sharding import DeviceMesh, DimSpec
from nabla.core.compute_graph import GRAPH


class TestUnevenSharding(unittest.IsolatedAsyncioTestCase):
    """Test sharding with tensor dims not evenly divisible by shard count."""
    
    def setUp(self):
        GRAPH._reset(None, 0)
        self.mesh = DeviceMesh("test", (4,), ("x",))
    
    async def test_prime_dimension_7(self):
        """7 elements across 4 shards: [2, 2, 2, 1] or clamped."""
        np_a = np.arange(7, dtype=np.float32).reshape(7, 1)
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Square each element
        result = a_sharded * a_sharded
        await result.realize
        
        expected = np_a * np_a
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)
    
    async def test_prime_dimension_11(self):
        """11 elements across 4 shards."""
        np_a = np.ones((11, 3), dtype=np.float32)
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = a_sharded + a_sharded
        await result.realize
        
        expected = np_a + np_a
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)
    
    async def test_fewer_elements_than_shards(self):
        """3 elements across 4 shards - some shards empty or shared."""
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1)
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = a_sharded * 2.0
        await result.realize
        
        expected = np_a * 2.0
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)


class TestSmallTensors(unittest.IsolatedAsyncioTestCase):
    """Test very small tensors that might have zero-size shards."""
    
    def setUp(self):
        GRAPH._reset(None, 0)
        self.mesh = DeviceMesh("test", (2,), ("x",))
    
    async def test_2x2_matmul(self):
        """Smallest useful matmul with sharding."""
        np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        a = Tensor.from_dlpack(np_a).shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        b = Tensor.from_dlpack(np_b)
        
        result = a @ b
        await result.realize
        
        expected = np_a @ np_b
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-4)
    
    async def test_1x1_tensor(self):
        """Single element tensor sharded."""
        np_a = np.array([[42.0]], dtype=np.float32)
        a = Tensor.from_dlpack(np_a).shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = a * 2.0
        await result.realize
        
        expected = np_a * 2.0
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)
    
    async def test_vector_1d(self):
        """1D vector sharded."""
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a = Tensor.from_dlpack(np_a).shard(self.mesh, [DimSpec(["x"])])
        
        result = a + 1.0
        await result.realize
        
        expected = np_a + 1.0
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)


class TestAsymmetricMeshes(unittest.IsolatedAsyncioTestCase):
    """Test asymmetric mesh configurations."""
    
    def setUp(self):
        GRAPH._reset(None, 0)
    
    async def test_2x4_mesh(self):
        """2x4 = 8 device mesh, shard on larger axis."""
        mesh = DeviceMesh("test", (2, 4), ("x", "y"))
        np_a = np.random.randn(16, 8).astype(np.float32)
        
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])
        
        result = a_sharded * 2.0
        await result.realize
        
        expected = np_a * 2.0
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)
    
    async def test_4x2_mesh(self):
        """4x2 = 8 device mesh, opposite orientation."""
        mesh = DeviceMesh("test", (4, 2), ("x", "y"))
        np_a = np.random.randn(8, 8).astype(np.float32)
        
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])
        
        result = a_sharded + a_sharded
        await result.realize
        
        expected = np_a + np_a
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)
    
    async def test_1x8_mesh_like_1d(self):
        """1x8 mesh should behave like 1D mesh with 8 devices."""
        mesh = DeviceMesh("test", (1, 8), ("x", "y"))
        np_a = np.random.randn(8, 4).astype(np.float32)
        
        a = Tensor.from_dlpack(np_a)
        # Shard only on y axis (the one with 8 devices)
        a_sharded = a.shard(mesh, [DimSpec(["y"]), DimSpec([])])
        
        result = a_sharded * 3.0
        await result.realize
        
        expected = np_a * 3.0
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)


class TestCommunicationOpsEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Edge cases for communication operations."""
    
    def setUp(self):
        GRAPH._reset(None, 0)
    
    async def test_gather_all_axes_2d_mesh(self):
        """gather_all_axes on tensor sharded across both mesh axes."""
        from nabla.ops.communication import gather_all_axes
        
        mesh = DeviceMesh("test", (2, 2), ("x", "y"))
        np_a = np.random.randn(4, 4).astype(np.float32)
        
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])
        
        gathered = gather_all_axes(a_sharded)
        await gathered.realize
        
        # Should reconstruct original
        np.testing.assert_allclose(gathered.to_numpy(), np_a, atol=1e-5)
    
    async def test_reduce_scatter_axis_0(self):
        """reduce_scatter along axis 0 instead of 1."""
        from nabla.ops.communication import reduce_scatter
        
        mesh = DeviceMesh("test", (4,), ("x",))
        np_a = np.ones((8, 4), dtype=np.float32)
        
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(mesh, [DimSpec([]), DimSpec(["x"])])
        
        result = reduce_scatter(a_sharded, axis=0)
        await result.realize
        
        # Sum of 4 shards, scattered to 8/4=2 rows per shard
        shard_shape = result._impl._values[0].type.shape if result._impl._values else result.shape
        self.assertEqual(int(shard_shape[0]), 2)  # 8 / 4 = 2
    
    async def test_allreduce_large_accumulation(self):
        """AllReduce with many values to test numerical precision.
        
        Input: (4, 100) sharded on axis 0 -> each shard is (1, 100)
        AllReduce sums all shards -> each shard still (1, 100) but replicated
        The value should be 4.0 (sum of 4 copies of 1.0)
        """
        from nabla.ops.communication import all_reduce
        
        mesh = DeviceMesh("test", (4,), ("x",))
        np_a = np.ones((4, 100), dtype=np.float32)
        
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = all_reduce(a_sharded)
        await result.realize
        
        # After allreduce on sharded axis, each shard has the same value (sum)
        # The result is replicated (all shards identical)
        # to_numpy() returns one shard since they're all identical
        # Each shard is (1, 100) and contains sum = 4.0
        result_np = result.to_numpy()
        self.assertEqual(result_np.shape, (1, 100))
        np.testing.assert_allclose(result_np, np.full((1, 100), 4.0), atol=1e-4)


class TestReductionEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Edge cases for reduction operations with sharding."""
    
    def setUp(self):
        GRAPH._reset(None, 0)
        self.mesh = DeviceMesh("test", (2,), ("x",))
    
    async def test_reduce_all_dims(self):
        """Reduce sum over all dimensions of sharded tensor."""
        from nabla import reduce_sum
        
        np_a = np.random.randn(4, 6).astype(np.float32)
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Reduce over both dims
        temp = reduce_sum(a_sharded, axis=1)
        result = reduce_sum(temp, axis=0)
        await result.realize
        
        expected = np.sum(np_a)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-4)
    
    async def test_reduce_keepdims_sharded(self):
        """Reduce with keepdims=True on sharded axis."""
        from nabla import reduce_sum
        
        np_a = np.random.randn(4, 6).astype(np.float32)
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = reduce_sum(a_sharded, axis=0, keepdims=True)
        await result.realize
        
        expected = np.sum(np_a, axis=0, keepdims=True)
        self.assertEqual(tuple(int(d) for d in result.shape), (1, 6))
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-4)


class TestBroadcastingWithSharding(unittest.IsolatedAsyncioTestCase):
    """Test broadcasting behavior with sharded tensors."""
    
    def setUp(self):
        GRAPH._reset(None, 0)
        self.mesh = DeviceMesh("test", (2,), ("x",))
    
    async def test_scalar_broadcast(self):
        """Broadcast scalar to sharded tensor."""
        np_a = np.random.randn(4, 4).astype(np.float32)
        a = Tensor.from_dlpack(np_a)
        a_sharded = a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = a_sharded + 5.0
        await result.realize
        
        expected = np_a + 5.0
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)
    
    async def test_row_broadcast_to_sharded(self):
        """Broadcast row vector to row-sharded matrix."""
        np_a = np.random.randn(4, 4).astype(np.float32)
        np_b = np.random.randn(1, 4).astype(np.float32)
        
        a = Tensor.from_dlpack(np_a).shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        b = Tensor.from_dlpack(np_b)
        
        result = a + b
        await result.realize
        
        expected = np_a + np_b
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
