
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


    def test_reshape_factor_propagation_and_reshard(self):
        """
        Test 'Shardy-like' factor propagation through reshapes, followed by restricted communication.
        
        Scenario:
        1. Input (4, 8) sharded on dim 0 (DP).
        2. Reshape to (32). DP factor is now on the single dimension.
        3. Reshape to (8, 4). 32 = 8 * 4.
           The logical mapping maps the original dim 0 (4, DP) to the NEW dim 1 (4).
           The original dim 1 (8, Replicated) maps to the NEW dim 0 (8).
           So output spec should be [{}, {"dp"}]. (Column sharded).
        4. Explicitly reshard to [{"dp"}, {}]. (Row sharded).
        5. This requires communication (All-to-All conceptually, or generic Reshard).
        """
        print("\n\n=== Trace: Reshape Factor Propagation + Reshard ===")
        
        dp = 2
        mesh = get_mesh((dp,), ("dp",))
        
        # Dimensions chosen to be distinct (4 vs 8) to verify factor tracking
        N, M = 4, 8
        x = Tensor.ones((N, M))
        # Shard x on dim 0: [dp, {}].
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([])])
        
        def complex_reshape_reshard(tensor):
            # 1. Start: [dp, {}] on (4, 8)
            
            # 2. Reshape -> (32).
            # Factor mapping: (4, 8) -> (32). The 32 dim contains both factor 4(dp) and factor 8.
            # Spec should show {"dp", ?}.
            flat = view.reshape(tensor, (N * M,))
            
            # 3. Reshape -> (8, 4). (M, N).
            # This effectively transposes the data layout logically by reshaping?
            # No, standard reshape is row-major.
            # (4, 8) flat is [row0, row1...].
            # (8, 4) flat is matching.
            # BUT the factors strictly track "which original dimension went where".
            # Original dim 0 (size 4) is the unit of periodicity?
            # 32 elements.
            # Original: 4 chunks of 8.
            # New: 8 chunks of 4.
            # The "inner" 4 elements of the new shape correspond to... simple chunks? NO.
            # 4 * 8:
            # Indices: (0,0)..(0,7), (1,0)..(1,7)...
            # Flat: 0..31.
            # New (8, 4):
            # Row 0 (0..3): Part of Old Row 0.
            # Row 1 (4..7): Part of Old Row 0.
            # Row 2 (8..11): Part of Old Row 1.
            # ...
            # Wait. Old Row 0 (size 8) contains 2 New Rows (size 4).
            # Old Row 0 was sharded on DP (device 0 has rows 0,1. device 1 has rows 2,3 - assuming dp=2).
            # Device 0 holds Flat indices 0..15.
            # Flat 0..15 corresponds to New Rows 0..3.
            # New Rows 0..3 correspond to New Shape (8, 4) indices (0,:)..(3,:).
            # So New Dim 0 (size 8) is split: Dev 0 has 0..3. Dev 1 has 4..7.
            # So New Dim 0 is sharded on DP!
            # New Dim 1 (size 4) is fully local on each device?
            # Let's see what the Propagation Engine decides!
            # If it says [dp, {}], my manual logic is right.
            # If it says [{}, dp], my "factor swap" hypothesis was right (transpose-like).
            # (N, M) -> (M, N) via reshape isn't transpose.
            swapped = view.reshape(flat, (M, N))

            # 4. Trigger Reshard to [{}, dp] (Force Column Sharding)
            # If propagation said [dp, {}], this is a Transpose-like reshard.
            target_spec = [DimSpec([]), DimSpec(["dp"])]
            resharded = swapped.shard(mesh, target_spec)
            
            return resharded

        trace = capture_trace(complex_reshape_reshard, x)
        print(trace)
        trace_str = str(trace)
        
        # Verify Reshapes present
        self.assertIn("reshape", trace_str)
        
        # Verify Reshard (communication) present
        # It usually appears as a 'copy' or 'shard' op in trace, or specialized collective?
        # GraphPrinter typically shows it.
        # But wait, shard() might not be a traceable op?
        # I added tracing to ShardOp in previous turn.
        # It should appear as: `d:f32[...] = shard c ...` or similar.
        # Searching for 'shard' inside the function body.
        
        # Extract the body of the trace
        self.assertIn("SHARD", trace_str)
        self.assertIn("ALL_GATHER", trace_str)
        
        # Verify sharding specs
        # We expect specs to appear.
        self.assertIn("@mesh", trace_str)
        
        # Numerical Verification
        import asyncio
        async def verify():
            resharded_tensor = trace.outputs if isinstance(trace.outputs, Tensor) else trace.outputs[0]
            await resharded_tensor.realize
            actual = resharded_tensor.to_numpy()
            
            # Expected
            np_x = np.ones((N, M), dtype=np.float32)
            # Just do the same reshapes
            np_flat = np_x.reshape(N * M)
            np_swapped = np_flat.reshape(M, N)
            
            np.testing.assert_allclose(actual, np_swapped)
            print(f"\n✓ Numerical verification PASSED: Shape {actual.shape}")
            
        asyncio.run(verify())

if __name__ == "__main__":
    unittest.main()
