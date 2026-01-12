
import asyncio
import unittest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.transforms.shard_map import shard_map
from nabla.core.trace import trace
from nabla import ops

class TestShardMapRigorous(unittest.TestCase):
    """Rigorous tests for shard_map verifying traces and execution."""

    def setUp(self):
        # Universal mesh for tests
        self.mesh = DeviceMesh("test_mesh", (2, 2), ("dp", "tp"), devices=[0, 1, 2, 3])
        print("\n" + "="*80)

    def tearDown(self):
        print("="*80 + "\n")

    def test_internal_constraint_trace(self):
        """Verify that internal constraints trigger implicit communication in the trace."""
        print("TEST: Internal Constraint Trace Analysis")
        print("-" * 60)
        
        # User function with manual constraint
        def func(x):
            # x starts replicated (implied)
            # Force sharding on 'dp' (dim 0)
            x = x.with_sharding_constraint(self.mesh, [DimSpec(["dp"]), DimSpec([])])
            y = x * 2
            # Force back to Replicated
            z = y.with_sharding_constraint(self.mesh, [DimSpec([]), DimSpec([])])
            return z

        # Wrap with shard_map: Input/Output replicated
        sharded_fn = shard_map(
            func, 
            self.mesh, 
            in_specs={0: None}, # Replicated
            out_specs=None      # Replicated output
        )

        # 1. Setup Input (Replicated)
        data = np.random.rand(4, 4).astype(np.float32)
        x = Tensor.from_dlpack(data)

        # 2. Trace the EXECUTION of the sharded function
        # This captures the replay ops
        print("Capturing Execution Trace...")
        t = trace(sharded_fn, x)
        print(t)
        
        # 3. Analyze Trace
        t_str = str(t)
        # We expect a 'shard' (or equivalent) op corresponding to the constraint
        # The constraint `x.with_sharding_constraint` results in `x.shard(...)` during replay.
        # This records a `shard` op in the trace.
        # And since it forces sharding, we might see `shard` ops.
        
        self.assertIn("shard", t_str)
        # We might check for specific sharding behaviors if visible in trace output
        
        # 4. Numerical Verification
        print("Running Numerical Verification...")
        result = sharded_fn(x)
        async def verify():
            # await result.realize # Optional if to_numpy handles it
            np_res = result.to_numpy()
            expected = data * 2
            np.testing.assert_allclose(np_res, expected)
            print("✅ PASS: Numerical verification")
        
        asyncio.run(verify())


    def test_io_constraints_sharding(self):
        """Test Input/Output constraints and verify shapes in trace."""
        print("TEST: I/O Constraints")
        print("-" * 60)
        
        # In: Distributed on 'dp', Out: Distributed on 'tp'
        in_spec = ShardingSpec(self.mesh, [DimSpec(["dp"]), DimSpec([])])
        out_spec = ShardingSpec(self.mesh, [DimSpec([]), DimSpec(["tp"])])

        def simple_add(a, b):
            return a + b

        sharded_fn = shard_map(
            simple_add,
            self.mesh,
            in_specs={0: in_spec, 1: in_spec}, # Both inputs distributed
            out_specs={0: out_spec}            # Output distributes on different axis
        )

        # Inputs
        d1 = np.random.rand(4, 4).astype(np.float32)
        d2 = np.random.rand(4, 4).astype(np.float32)
        t1 = Tensor.from_dlpack(d1) # Start replicated
        t2 = Tensor.from_dlpack(d2)

        # Trace
        print("Capturing Execution Trace...")
        t = trace(sharded_fn, t1, t2)
        print(t)

        # Analysis
        # The trace should show:
        # 1. Inputs being sharded (implicitly or explicitly via shard_map preamble)
        # 2. Add op
        # 3. Output being sharded to 'tp'
        
        # Run
        result = sharded_fn(t1, t2)
        
        async def verify():
            print(f"Result Sharding: {result.sharding}")
            # Verify output spec
            self.assertEqual(result.sharding.dim_specs[1].axes, ["tp"])
            
            np_res = result.to_numpy()
            expected = d1 + d2
            np.testing.assert_allclose(np_res, expected, atol=1e-5)
            print("✅ PASS: Result Verified")
            
        asyncio.run(verify())

    def test_complex_graph_trace(self):
        """Verify complex graph with mixed constraints."""
        print("TEST: Complex Graph Trace")
        print("-" * 60)

         # x: [B, H], w1: [H, 4H], w2: [4H, H]
        def complex_func(x, w1, w2):
            h = x @ w1
            h = h * 2.0
            # Force intermediate hidden dim on 'tp'
            h = h.with_sharding_constraint(self.mesh, [DimSpec([]), DimSpec(["tp"])])
            out = h @ w2
            return out

        sharded_fn = shard_map(
            complex_func,
            self.mesh,
            in_specs={0: None, 1: None, 2: None},
            out_specs={0: None}
        )
        
        B, H = 4, 4
        d_x = np.random.rand(B, H).astype(np.float32)
        d_w1 = np.random.rand(H, 4*H).astype(np.float32)
        d_w2 = np.random.rand(4*H, H).astype(np.float32)
        
        t_x = Tensor.from_dlpack(d_x)
        t_w1 = Tensor.from_dlpack(d_w1)
        t_w2 = Tensor.from_dlpack(d_w2)
        
        print("Capturing Trace...")
        t = trace(sharded_fn, t_x, t_w1, t_w2)
        print(t)
        
        # The trace should reveal the structure.
        # We look for the 'shard' op in the middle corresponding to the constraint.
        t_str = str(t)
        self.assertIn("shard", t_str)
        
        async def verify():
            result = sharded_fn(t_x, t_w1, t_w2)
            res_np = result.to_numpy()
            expected = (d_x @ d_w1 * 2.0) @ d_w2
            np.testing.assert_allclose(res_np, expected, atol=1e-4)
            print("✅ PASS: Complex Graph Verified")
            
        asyncio.run(verify())

if __name__ == "__main__":
    unittest.main(verbosity=2)
