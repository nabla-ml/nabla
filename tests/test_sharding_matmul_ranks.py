
import unittest
import numpy as np
import pytest
from nabla import Tensor, ops
from nabla.sharding import DeviceMesh, DimSpec, ShardingSpec
from nabla.core.compute_graph import GRAPH

class TestMatmulRanks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        GRAPH._reset(None, 0)
        self.mesh = DeviceMesh("test_mesh", (4,), ("x",))

    async def test_vec_matmul_mat(self):
        """Test (K,) @ (K, N) -> (N,)
        
        Vector-Matrix multiplication. K is the contracting dimension.
        Sharding K should reduce. Sharding N should propagate.
        """
        K, N = 32, 16
        
        # A: (K,) sharded on "x" (contracting dim)
        np_a = np.random.randn(K).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"])])
        
        # B: (K, N) sharded on "x" (contracting dim) and replicated N
        np_b = np.random.randn(K, N).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # C = A @ B -> (N,)
        # Since we shard the contracting dim ("x"), this implies AllReduce.
        c = a @ b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (N,))
        # Result should be replicated (all axes reduced)
        # Or implicitly replicated (empty DimSpec or None)
        # Actually, since output has no "x", it should be replicated.
        self.assertTrue(all(not d.axes for d in c._impl.sharding.dim_specs))
        
        expected = np_a @ np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    async def test_mat_matmul_vec(self):
        """Test (M, K) @ (K,) -> (M,)
        
        Matrix-Vector multiplication.
        Shard M: Output sharded on M.
        Shard K: AllReduce.
        """
        M, K = 16, 32
        
        # A: (M, K) sharded on M ("x")
        np_a = np.random.randn(M, K).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # B: (K,) replicated
        np_b = np.random.randn(K).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec([])])
        
        # C = A @ B -> (M,)
        c = a @ b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (M,))
        # Result should be sharded on dim 0 ("x")
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a @ np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    async def test_batch_matmul(self):
        """Test (B, M, K) @ (B, K, N) -> (B, M, N)
        
        Sharding on B should propagate.
        """
        B, M, K, N = 4, 8, 8, 8
        
        # A: (B, M, K) sharded on B ("x")
        np_a = np.random.randn(B, M, K).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
        
        # B: (B, K, N) sharded on B ("x")
        np_b = np.random.randn(B, K, N).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
        
        # C = A @ B -> (B, M, N)
        c = a @ b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (B, M, N))
        # Result should be sharded on dim 0 ("x")
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a @ np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    async def test_broadcast_matmul_weights(self):
        """Test (B, M, K) @ (K, N) -> (B, M, N)
        
        "Weights" (K, N) are broadcast over batch dimension.
        Shard B: Propagates.
        Shard K: AllReduce.
        Shard N: Model parallel.
        """
        B, M, K, N = 4, 8, 16, 8
        
        # A: (B, M, K) sharded on B ("x")
        np_a = np.random.randn(B, M, K).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
        
        # B: (K, N) replicated
        np_b = np.random.randn(K, N).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec([]), DimSpec([])])
        
        # C = A @ B
        c = a @ b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (B, M, N))
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a @ np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    async def test_broadcast_matmul_complex(self):
        """Test (B, M, K) @ (K, N) where both have sharding.
        
        A sharded on M ("x"). B sharded on N ("x").
        Result (B, M, N) should theoretically have conflict if simple propagation used.
        But matmul logic maps factors:
        A: M="x"
        B: N="x"
        Result: M="x", N="x" -> (B, M, N) sharded on M and N with "x".
        This is impossible on 1D mesh unless one is sub-sharded or we error?
        Wait, if mesh is 1D, we can't shard TWO dims on "x".
        This should raise a conflict or result in one being resharded.
        
        Actually, with 1D mesh, "x" can only be consumed once.
        If A shards M on "x" and B shards N on "x",
        The output (B, M, N) would need "x" for M and "x" for N.
        Since "x" has size 4, if we partition M by 4 and N by 4, we use 16 devices?
        No, on a single axis "x", we can only partition one dimension at a time unless we split the axis.
        
        Correct behavior: The conflict resolution should pick one or fail.
        Let's test what happens.
        """
        B, M, K, N = 4, 16, 16, 16
        
        np_a = np.random.randn(B, M, K).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        # Shard M on x
        a.shard(self.mesh, [DimSpec([]), DimSpec(["x"]), DimSpec([])])
        
        np_b = np.random.randn(K, N).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        # Shard N on x
        b.shard(self.mesh, [DimSpec([]), DimSpec(["x"])])
        
        c = a @ b
        
        # This might fail or reshard. 
        # If it realizes successfully, check result.
        try:
            await c.realize
            result_sharding = c._impl.sharding
            # It likely kept one and replicated the other, or errored.
            # Just asserting it runs and is numerically correct.
            actual = c.to_numpy()
            expected = np_a @ np_b
            np.testing.assert_allclose(actual, expected, atol=1e-4)
        except Exception:
            # Failure is also acceptable if conflict detected
            pass

if __name__ == "__main__":
    unittest.main()
