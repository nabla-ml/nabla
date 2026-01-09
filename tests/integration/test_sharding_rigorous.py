
import unittest
import numpy as np
import nabla
from nabla import ops
from nabla.core.trace import trace
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec

class TestShardingRigorous(unittest.TestCase):
    def setUp(self):
        # Create a standard mesh for testing
        self.mesh = DeviceMesh(name="mesh", shape=(2, 2), axis_names=("x", "y"))
        
    def test_concat_sharding_trace(self):
        """Verify ConcatenateOp sharding correctly implies bidirectional propagation (Forward check)."""
        def func(a, b):
            # Shard inputs on 'x'
            a = ops.shard(a, self.mesh, [DimSpec(["x"]), DimSpec([])])
            b = ops.shard(b, self.mesh, [DimSpec(["x"]), DimSpec([])])
            
            # Concatenate along axis 0
            # Shared factor 'c_concat' should pick up 'x' from inputs.
            c = ops.concatenate([a, b], axis=0)
            return c

        a = nabla.Tensor.zeros((4, 4))
        b = nabla.Tensor.zeros((4, 4))
        
        traced_output = str(trace(func, a, b))
        
        print("\n--- Trace Output (Concat) ---\n")
        print(traced_output)
        
        # Check that output sharding spec contains "x"
        # We look for %v... = concatenate...
        # %v... (<x, *>)
        # The output variable from concatenate should have the spec.
        self.assertIn("(<x, *>)", traced_output)
        
        # Verify inputs were sharded (showing correct trace structure)
        # We search for shard op being applied to argument %a1
        # Trace has ANSI colors, so we check loosely or use regex
        self.assertRegex(traced_output, r"shard.*%a1")
        self.assertRegex(traced_output, r"shard.*%a2")
        
    def test_reduce_sharding_trace(self):
        """Verify Reduce sharding rules for keepdims=True/False."""
        def func_keep(x):
            # Shard input on 'x'
            x = ops.shard(x, self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
            return ops.reduce_sum(x, axis=1, keepdims=True)
            
        def func_drop(x):
            x = ops.shard(x, self.mesh, [DimSpec(["x"]), DimSpec(["y"])])
            return ops.reduce_sum(x, axis=1, keepdims=False)
            
        x = nabla.Tensor.zeros((4, 4))
        
        trace_keep = str(trace(func_keep, x))
        trace_drop = str(trace(func_drop, x))
        
        print("\n--- Trace Output (Reduce Keep) ---\n")
        print(trace_keep)
        # Output sharding should be (<x, *>) -- the kept-dim is empty factor (replicated/none)
        # The Trace printer shows (*) for empty factors.
        # So we expect (<x, *>) on the output variable.
        self.assertIn("(<x, *>)", trace_keep)
        
        print("\n--- Trace Output (Reduce Drop) ---\n")
        print(trace_drop)
        # Output sharding should be (<x>) (rank 1)
        self.assertIn("(<x>)", trace_drop)
        
    def test_mlp_sharding_trace(self):
        """End-to-End MLP sharding verification."""
        def mlp(x, w1, w2):
            # x: (B, H) -> Sharded on 'x' (DP)
            # w1: (H, D) -> Sharded on 'y' (TP)
            # w2: (D, H) -> Sharded on 'y' (TP)
            
            x = ops.shard(x, self.mesh, [DimSpec(["x"]), DimSpec([])])
            w1 = ops.shard(w1, self.mesh, [DimSpec([]), DimSpec(["y"])])
            w2 = ops.shard(w2, self.mesh, [DimSpec(["y"]), DimSpec([])])
            
            # Matmul 1: (B, H) @ (H, D) -> (B, D)
            # Factors: d0=B (sharded x), d1=H (replicated in x, but w1 has H? No w1 has H at dim 0)
            # Wait. x: dim 1 is H. w1: dim 0 is H.
            # x has H replicated. w1 has H replicated. Good.
            # w1 has D sharded on y. x has B sharded on x.
            # Result h1: (B, D) -> Sharded on x (from B) and y (from D). -> [x, y]
            h1 = x @ w1
            h1 = ops.relu(h1)
            
            # Matmul 2: (B, D) @ (D, H) -> (B, H)
            # h1: (B, D) [x, y]. w2: (D, H) [y, ?]
            # Contraction on D (dim 1 of h1, dim 0 of w2).
            # h1 has D sharded on 'y'. w2 has D sharded on 'y'. Match!
            # Result logits: (B, H) -> [x, ?] (from B and H)
            # w2 H is dim 1. w2[dim1] is replicated.
            # So output should be [x, ?] (replicated H).
            # But D was sharded on 'y', so we summed over 'y'.
            # Trace should show AllReduce on 'y' for the output? 
            # Or does matmul rule imply output inherits 'y'? 
            # No, 'y' is on contracting dimension D. It disappears from output factors.
            # So output is Partial-SUM on 'y'.
            # Then we might need AllReduce(y) to get Fully Replicated result on y.
            logits = h1 @ w2
            return logits

        x = nabla.Tensor.zeros((16, 32))
        w1 = nabla.Tensor.zeros((32, 64))
        w2 = nabla.Tensor.zeros((64, 32))
        
        traced = str(trace(mlp, x, w1, w2))
        print("\n--- Trace Output (MLP) ---\n")
        print(traced)
        
        # Verify basic structural properties of the trace
        self.assertIn("matmul", traced)
        self.assertIn("relu", traced)
        
        # Verify AllReduce insertion
        self.assertIn("all_reduce", traced)
        
        # Verify sharding specs
        # Matmul 1 output: (<x, y>)
        self.assertIn("(<x, y>)", traced)
        # Final output prior to AllReduce: (<x, *>) (Partial sum on y)
        # Final output after AllReduce: (<x, *>) (Replicated on y)
        self.assertIn("(<x, *>)", traced)

if __name__ == "__main__":
    unittest.main()
