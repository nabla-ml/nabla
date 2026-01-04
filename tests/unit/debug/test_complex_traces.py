
import os
import sys
import unittest
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec
from nabla.utils.debug import capture_trace, xpr
from nabla.ops import view, creation
from nabla.transforms import vmap
import nabla.ops as ops

def get_mesh(shape, axes):
    return DeviceMesh("mesh", shape, axes, devices=list(range(np.prod(shape))))

class TestComplexTraces(unittest.TestCase):
    
    def test_sharded_attention_split(self):
        """
        Test sharding propagation through Reshape and Transpose (Split Heads).
        This mimics a Transformer Attention layer input transformation.
        
        Transformation:
        (Batch, Seq, D_Model) -> (Batch, Seq, Heads, Head_Dim) -> (Batch, Heads, Seq, Head_Dim)
        
        Sharding:
        Batch: DP
        D_Model: TP
        """
        print("\n\n=== Trace: Sharded Attention Split (Reshape + Transpose) ===")
        
        dp, tp = 2, 2
        mesh = get_mesh((dp, tp), ("dp", "tp"))
        
        B, S, H, D = 4, 8, 4, 16 # Heads=4, Head_Dim=16 -> D_Model=64
        D_Model = H * D
        
        # Input sharded on DP (Batch) and TP (D_Model)
        x = Tensor.ones((B, S, D_Model))
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([]), DimSpec(["tp"])])
        
        def split_heads(hidden_states):
            # Reshape: (B, S, D_Model) -> (B, S, H, D)
            # TP is on D_Model. Reshape splits D_Model -> (H, D).
            reshaped = view.reshape(hidden_states, (B, S, H, D))
            
            # Transpose: (B, S, H, D) -> (B, H, S, D)
            transposed = view.swap_axes(reshaped, 1, 2)
            return transposed

        trace = capture_trace(split_heads, x)
        print(trace)
        
        # Verification Logic
        # We process the trace string to verify specific output patterns
        trace_str = str(trace)
        
        # 1. Verify Reshape Output Sharding
        # 2. Verify Output Shape
        # 'xpr' shows output usage LOCAL shapes (shards).
        # Global: (4, 4, 8, 16).
        # Local:  (2, 2, 8, 16) (Batch/2, Heads/2).
        self.assertIn(f"f32[{int(B/dp)}, {int(H/tp)}, {S}, {D}]", trace_str)

        # 3. Numerical Verification
        import asyncio
        async def verify():
            # Get result from trace outputs
            result_tensor = trace.outputs if isinstance(trace.outputs, Tensor) else trace.outputs[0]
            
            # Trigger execution
            await result_tensor.realize
            
            # Get actual values
            actual = result_tensor.to_numpy()
            
            # Compute expected with NumPy
            np_x = np.ones((B, S, D_Model), dtype=np.float32)
            np_reshaped = np_x.reshape(B, S, H, D)
            # Swap axes 1 and 2: (B, S, H, D) -> (B, H, S, D)
            np_transposed = np.swapaxes(np_reshaped, 1, 2)
            
            np.testing.assert_allclose(actual, np_transposed)
            print(f"\n✓ Numerical verification PASSED: Shape {actual.shape}")
            
        asyncio.run(verify())


    def test_nested_vmap_reduction(self):
        """
        Test tracing of nested vmap with reduction.
        Shows how batch dims stack and how reduction axis is handled relative to them.
        """
        print("\n\n=== Trace: Nested Vmap Reduction (Norm) ===")
        
        # Function: compute L2 norm of a vector (simulated with exp for tracing)
        def vec_norm(v):
            sq = ops.mul(v, v)
            s = ops.reduce_sum(sq, axis=0) # Reduce vector dim
            return ops.exp(s)
        
        # vmap over batch of vectors
        batch_norm = vmap(vec_norm)
        
        # vmap over batch of batches (e.g. [Batch, Groups, VecDim])
        nested_norm = vmap(batch_norm)
        
        B1, B2, V = 2, 3, 4
        x = Tensor.ones((B1, B2, V))
        
        # Capture trace
        trace = capture_trace(nested_norm, x)
        result_tensor = trace.outputs[0] if isinstance(trace.outputs, list) else trace.outputs
        
        print(trace)
        trace_str = str(trace)
        
        # Verify trace structure
        self.assertIn("reduce_sum", trace_str)
        self.assertIn("exp", trace_str)
        self.assertIn("incr_batch_dims", trace_str)
        
        # Numerical Verification
        import asyncio
        async def verify():
            await result_tensor.realize
            actual = result_tensor.to_numpy()
            
            # Compute expected with NumPy
            np_x = np.ones((B1, B2, V), dtype=np.float32)
            # vec_norm: exp(sum(v*v))
            # inner vmap maps over V (dim 2). 
            # Wait, vmap maps over leading dim by default?
            # vmap(f) -> f takes x[i].
            # batch_norm = vmap(vec_norm). Input (B2, V). 
            # vec_norm takes (V,). reduces axis 0 (V).
            # So batch_norm takes (B2, V). Returns (B2,).
            # nested_norm = vmap(batch_norm). Input (B1, B2, V).
            # Returns (B1, B2).
            
            # Numpy equivalent:
            # sq = x * x
            # sum = np.sum(sq, axis=2) # axis 2 is V
            # res = np.exp(sum)
            
            np_sq = np_x * np_x
            np_sum = np.sum(np_sq, axis=2)
            np_res = np.exp(np_sum)
            
            np.testing.assert_allclose(actual, np_res, rtol=1e-5)
            print(f"\n✓ Numerical verification PASSED: Shape {actual.shape}")
            
        asyncio.run(verify())

if __name__ == "__main__":
    unittest.main()
