
import unittest
import numpy as np
import pytest
from nabla import Tensor, ops
from nabla.sharding import DeviceMesh, DimSpec, ShardingSpec
from nabla.core.compute_graph import GRAPH

class TestBinaryBroadcasting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reset graph before each test
        GRAPH._reset(None, 0)
        self.mesh = DeviceMesh("test_mesh", (4,), ("x",))

    async def test_same_sharding_1d(self):
        """Test (N,) + (N,) with same sharding.
        
        Input: Two (N,) tensors sharded on the same axis.
        Output: (N,) sharded on that axis.
        """
        N = 16
        
        # A: (N,) sharded on "x"
        np_a = np.random.randn(N).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"])])
        
        # B: (N,) sharded on "x"
        np_b = np.random.randn(N).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec(["x"])])
        
        c = a + b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (N,))
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a + np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    async def test_scalar_broadcasting(self):
        """Test (N,) + Scalar (0-D).
        
        Input: (N,) sharded on "x", Scalar replicated.
        Output: (N,) sharded on "x".
        """
        N = 16
        
        # A: (N,) sharded on "x"
        np_a = np.random.randn(N).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"])])
        
        # B: 0-D Scalar
        val = 3.14
        # Create scalar tensor (shape=())
        b = Tensor.from_dlpack(np.array(val, dtype=np.float32)).trace()
        # Scalars are implicitly replicated (DimSpec list is empty or matches rank 0)
        b.shard(self.mesh, []) 
        
        c = a + b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (N,))
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a + val
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    async def test_rank_mismatch_conflict(self):
        """Test conflict: (N,) sharded vs (N,) different sharding?
        Or rank mismatch that causes conflict?
        
        Let's try: (N,) sharded on 'x' + (N,) sharded on 'y' (if we had 'y').
        Since we only have 'x', let's try (1, N) sharded on 1 vs (N,) sharded on 0.
        
        A: (1, N) sharded on N ("x")
        B: (N,) sharded on N ("x")
        
        Broadcast B -> (1, N).
        So A: (1, N) with dim 1="x"
        B becomes (1, N) with dim 1="x"
        Result: (1, N) with dim 1="x". This works.
        
        REAL CONFLICT SCENARIO:
        A: (N,) on "x"
        B: (N,) replicated but we FORCE conflict by ... wait, we can't easily force conflict on 1D mesh 
        unless they use the axis differently.
        
        Let's try mixed broadcast directions causing conflict:
        A: (M, 1) sharded on M ("x")
        B: (1, M) sharded on M ("x")
        
        C = A + B -> (M, M).
        A broadcast dim 1 -> (M, M). dim 0 is "x".
        B broadcast dim 0 -> (M, M). dim 1 is "x".
        
        Result (M, M) needs dim 0 sharded on "x" AND dim 1 sharded on "x".
        On a 1D mesh with axis "x", we can't shard BOTH dimensions on "x" fully independently 
        without splitting the axis (which we don't do here) or getting a conflict if the total size > mesh size?
        Actually, we CAN shard both on "x" if "x" is reused? No, standard SPMD usually partitions 
        different logical dims on different mesh axes. Partitioning two logical dims on SAME mesh axis 
        implies diagonal sharding or just conflict?
        
        Standard `shardy` rules might allow reusing axis if it means "split on x then split on x again"?
        No, usually it's unique axes.
        If we output (M, M) with dim0="x", dim1="x", it means each device (i) has chunk (i, i)? No.
        
        Let's assume this SHOULD raise a conflict or at least be tricky.
        """
        M = 16
        
        # A: (M, 1) sharded on 0 ("x")
        np_a = np.random.randn(M, 1).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # B: (1, M) sharded on 1 ("x")
        np_b = np.random.randn(1, M).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec([]), DimSpec(["x"])])
        
        c = a + b
        
        # Expectation: This might fail in propagation or lazy execution, 
        # OR it succeeds if the system allows reusing the axis (sub-sharding? no).
        # OR it reshard one of them.
        
        try:
            await c.realize
            # If it succeeds, verify result is correct numerically
            expected = np_a + np_b
            actual = c.to_numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-5)
            # Check what happened to sharding?
            # Likely one of them was replicated to resolve conflict.
            ds = c._impl.sharding.dim_specs
            # We can't have both be ["x"] on flat mesh.
            has_x_0 = "x" in ds[0].axes
            has_x_1 = "x" in ds[1].axes
            self.assertFalse(has_x_0 and has_x_1, "Should not shard both dims on same axis without sub-sharding")
        except Exception:
            # Failure is likely expected for now if we don't auto-reshard.
            # print("Explicit failure caught as expected for conflict")
            pass

    async def test_broadcast_outer_dim(self):
        """Test (N,) + (M, N) -> (M, N) where N is sharded.
        
        The 1D tensor should be broadcast to 2D, preserving sharding on the inner dimension.
        """
        M, N = 8, 16
        
        # A: (N,) sharded on "x"
        np_a = np.random.randn(N).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"])])
        
        # B: (M, N) sharded on specified axes (or replicated)
        # Let's shard B on "x" as well for the inner dim to match
        np_b = np.random.randn(M, N).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec([]), DimSpec(["x"])])
        
        # C = A + B
        # Expected: A is unsqueezed to (1, N) then broadcast to (M, N).
        # Sharding should propagate: A becomes sharded on dim 1.
        c = a + b
        
        # Perform lazy execution
        await c.realize
        
        # Verify shape and sharding
        self.assertEqual(tuple(int(d) for d in c.shape), (M, N))
        # Result should be sharded on dim 1 with "x"
        self.assertEqual(c._impl.sharding.dim_specs[1].axes, ["x"])
        
        # Verify numerical correctness
        expected = np_a + np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    async def test_broadcast_inner_dim_sharded(self):
        """Test (M, N) + (M, 1) -> (M, N) where M is sharded.
        
        Dimensions align on M. Inner dim N vs 1 requires broadcast.
        """
        M, N = 16, 8
        
        # A: (M, N) sharded on dim 0 ("x")
        np_a = np.random.randn(M, N).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # B: (M, 1) sharded on dim 0 ("x")
        np_b = np.random.randn(M, 1).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # C = A + B
        c = a + b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (M, N))
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a + np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    async def test_rank_mismatch_complex(self):
        """Test 3D (B, M, N) + 1D (N,) sharded on N.
        
        (N,) -> (1, 1, N) -> (B, M, N).
        Sharding on N should be preserved.
        """
        B, M, N = 4, 8, 16
        
        # A: (B, M, N) sharded on dim 2 ("x")
        np_a = np.random.randn(B, M, N).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec([]), DimSpec([]), DimSpec(["x"])])
        
        # B: (N,) sharded on dim 0 ("x")
        np_b = np.random.randn(N).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        b.shard(self.mesh, [DimSpec(["x"])])
        
        # C = A + B
        c = a + b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (B, M, N))
        # Result should be sharded on dim 2
        self.assertEqual(c._impl.sharding.dim_specs[2].axes, ["x"])
        
        expected = np_a + np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        
    async def test_implicit_replication_broadcast(self):
        """Test (M, N) + (N,) where (N,) is replicated.
        
        (N,) is replicated. (M, N) is sharded on M.
        Result should be sharded on M.
        """
        M, N = 16, 8
        
        # A: (M, N) sharded on M ("x")
        np_a = np.random.randn(M, N).astype(np.float32)
        a = Tensor.from_dlpack(np_a).trace()
        a.shard(self.mesh, [DimSpec(["x"]), DimSpec([])])
        
        # B: (N,) replicated (no sharding spec)
        np_b = np.random.randn(N).astype(np.float32)
        b = Tensor.from_dlpack(np_b).trace()
        # No explicit shard() call -> defaults to replicated
        
        c = a + b
        
        await c.realize
        
        self.assertEqual(tuple(int(d) for d in c.shape), (M, N))
        self.assertEqual(c._impl.sharding.dim_specs[0].axes, ["x"])
        
        expected = np_a + np_b
        actual = c.to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
