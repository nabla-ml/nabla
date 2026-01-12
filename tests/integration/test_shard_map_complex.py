# ===----------------------------------------------------------------------=== #
# Nabla 2026 - Complex Shard Map Integration Tests
# ===----------------------------------------------------------------------=== #

import unittest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.core.trace import trace
from nabla.transforms.shard_map import shard_map
from nabla.sharding.spec import DeviceMesh, ShardingSpec, DimSpec
from nabla.ops import binary, view, reduction

class TestShardMapComplex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a mock mesh for testing
        cls.mesh_shape = (2, 2)
        cls.device_ids = [0, 1, 2, 3]
        cls.axis_names = ("x", "y")
        cls.mesh = DeviceMesh("test_mesh", cls.mesh_shape, cls.axis_names, devices=cls.device_ids)

    def test_reduction_on_sharded_axis(self):
        """Test reducing a dimension that is sharded."""
        # Setup: Input (4, 4) sharded on 'x' (axis 0) -> (2, 4) per shard
        # Reduce sum on axis 0 -> Output (4,) replicated? Or partial? 
        # ShardMap logic should delegate to reduction op, which should return a tensor 
        # suitable for the output spec.
        
        # NOTE: standard reduction across a sharded axis usually results in replication 
        # (all-reduce) unless specified otherwise.
        
        in_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])}
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec([])])} # Output is replicated

        def my_reduce(x):
            return reduction.reduce_sum(x, axis=0)

        # Logical Input: (4, 4)
        x_np = np.ones((4, 4), dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        sharded_fn = shard_map(my_reduce, self.mesh, in_specs, out_specs)
        
        # Execute
        res = sharded_fn(x)
        res._sync_realize()

        # Verification
        expected = np.sum(x_np, axis=0) # (4,)
        np.testing.assert_allclose(res.to_numpy(), expected, rtol=1e-5)
        
        # Check internal dual state (sanity check, user shouldn't see this)
        # The result should be Dual-backed during execution.
    
    def test_reshape_resharding(self):
        """Test reshape that might imply sharding changes."""
        # Input (4, 4) sharded on x (axis 0).
        # Reshape to (16,) -> How does sharding propagate? 
        # Typically reshape preserves total size.
        
        in_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])}
        # Output sharded on x? (16,) splits into (8,) per shard on axis x
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"])])}

        def my_reshape(x):
            return view.reshape(x, (16,))

        x_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        x = Tensor.from_dlpack(x_np)

        sharded_fn = shard_map(my_reshape, self.mesh, in_specs, out_specs)
        
        res = sharded_fn(x)
        res._sync_realize()
        
        expected = x_np.reshape(16)
        np.testing.assert_allclose(res.to_numpy(), expected, rtol=1e-5)

    def test_matmul_contraction(self):
        """Test matmul where the contracting dimension is sharded."""
        # (M, K) @ (K, N) -> (M, N)
        # Shard K on 'x'. This requires all-reduce on the output.
        
        M, K, N = 4, 8, 4
        
        # A: (M, K) sharded on 'x' along K (axis 1)
        # B: (K, N) sharded on 'x' along K (axis 0)
        in_specs = {
            0: ShardingSpec(self.mesh, [DimSpec([]), DimSpec(["x"])]),
            1: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])
        }
        # Output: (M, N) replicated (since we sum over sharded K)
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec([]), DimSpec([])])}

        def my_matmul(a, b):
            return binary.matmul(a, b)

        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)
        
        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)

        sharded_fn = shard_map(my_matmul, self.mesh, in_specs, out_specs)
        
        res = sharded_fn(a, b)
        res._sync_realize()
        
        expected = a_np @ b_np
        np.testing.assert_allclose(res.to_numpy(), expected, rtol=1e-4)

    def test_complex_composition_and_constants(self):
        """Test a mix of ops, broadcasting with constants, and reductions."""
        
        # Input: X (4, 4)
        # 1. Add scalar constant (broadcast)
        # 2. Matmul with generic constant weight
        # 3. Reduce sum
        
        in_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])}
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec([])])}
        
        W_np = np.eye(4, dtype=np.float32) * 2.0
        
        def my_model(x):
            # x is sharded
            h1 = binary.add(x, 1.0) # implicit constant broadcast
            W = Tensor.from_dlpack(W_np) # Constant tensor input (captured as dual=None fallback?)
            # NOTE: If we define W inside, it's a new tensor every time. 
            # Ideally W should be an argument if we want it sharded. 
            # But here we test "constant fallback" logic in shard_map trace.
            
            h2 = binary.matmul(h1, W)
            return reduction.reduce_sum(h2, axis=1)

        x_np = np.ones((4, 4), dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        sharded_fn = shard_map(my_model, self.mesh, in_specs, out_specs)
        
        res = sharded_fn(x)
        res._sync_realize()
        
        # Expected
        # (x + 1) @ W -> (2 @ 2I) = 4
        # sum(axis=1) -> [16, 16, 16, 16]
        expected = np.sum((x_np + 1.0) @ W_np, axis=1)
        np.testing.assert_allclose(res.to_numpy(), expected, rtol=1e-5)

    def test_output_spec_enforcement(self):
        """Test that out_specs forces resharding if result doesn't match."""
        # Function: x -> x (Identity)
        # Input: Replicated
        # Output Spec: Sharded on 'x'
        # Expectation: Result should be sharded.

        in_specs = {0: ShardingSpec(self.mesh, [DimSpec([]), DimSpec([])])} # Replicated input
        out_specs = {0: ShardingSpec(self.mesh, [DimSpec(["x"]), DimSpec([])])} # Sharded output

        def identity(x):
            return x

        x_np = np.ones((4, 4), dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        sharded_fn = shard_map(identity, self.mesh, in_specs, out_specs)
        
        # Run tracing
        res = sharded_fn(x)
        
        # Verify result IS sharded (it is the physical tensor)
        self.assertIsNotNone(res._impl.sharding)
        self.assertEqual(res._impl.sharding.dim_specs[0].axes, ["x"])
        self.assertEqual(res._impl.sharding.dim_specs[1].axes, [])
        
        # Run it
        res._sync_realize()
        expected = x_np
        np.testing.assert_allclose(res.to_numpy(), expected)

if __name__ == "__main__":
    unittest.main()
