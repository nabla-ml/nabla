
import unittest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec
from nabla.utils import debug
import nabla.ops as ops
from nabla.ops import view, creation

def get_mesh(shape, axes):
    return DeviceMesh("cluster", shape, axes, devices=list(range(np.prod(shape))))

def softmax(x, axis=-1):
    e = ops.exp(x)
    s = ops.reduce_sum(e, axis=axis, keepdims=True)
    return ops.div(e, s)

class TestSequenceShardedAttention(unittest.TestCase):
    
    def test_sequence_conflict_trace(self):
        """
        Test sharding propagation forcing a conflict on the Sequence dimension.
        Target: Q and K both sharded on Sequence (TP).
        MatMul (Q @ K.T) will request TP on both output dimensions, forcing resolution.
        """
        print("\n\n================================================================================")
        print("TRACE: Sequence Sharded Attention (Forced Conflict)")
        print("================================================================================")
        
        # 1. Setup Mesh (2, 2)
        dp, tp = 2, 2
        mesh = get_mesh((dp, tp), ("dp", "tp"))
        
        # 2. Dimensions
        B, S = 2, 8
        Heads, HeadDim = 2, 16
        
        # 3. Tensors
        # Input X: [B, S, H, D]
        # Sharded: [dp, tp, *, *] -> Sequence (axis 1) is sharded on TP!
        # Use small values to prevent softmax overflow
        X = (Tensor.ones((B, S, Heads, HeadDim)) * 0.1).shard(mesh, [DimSpec(["dp"]), DimSpec(["tp"]), DimSpec([]), DimSpec([])])
        
        # Weights (Local / Replicated for simplicity)
        # Using identity projection concept (skip weights) to isolate the sharding logic
        # OR just treat X as Q and K directly.
        
        def attention_op(x):
            # x: [B, S, H, D] <dp, tp, *, *>
            
            # 1. Prepare Q: Swap to [B, H, S, D]
            # Output Sharding: [dp, *, tp, *]
            q = view.swap_axes(x, 1, 2)
            
            # 2. Prepare K.T: Swap to [B, H, D, S]
            # First swap H, S -> [B, H, S, D] <dp, *, tp, *>
            k_temp = view.swap_axes(x, 1, 2)
            # Then swap S, D -> [B, H, D, S] <dp, *, *, tp>
            k_t = view.swap_axes(k_temp, 2, 3)
            
            # 3. Scores = Q @ K.T
            # Q:   [B, H, S, D] <dp, *, tp, *>
            # K.T: [B, H, D, S] <dp, *, *, tp>
            # MatMul Output: [B, H, S, S]
            # Propagated constraints:
            # - Axis 2 (S from Q): wants <tp>
            # - Axis 3 (S from K.T): wants <tp>
            # CONFLICT! Can't have <dp, *, tp, tp> on 1D 'tp' mesh.
            # System must insert AllGather on one input.
            scores = ops.matmul(q, k_t)
            
            # 4. Softmax
            # Axis -1 is S.
            # If K was AllGathered, axis -1 might be replicated.
            # If Q was AllGathered, axis -1 might be sharded (tp).
            # Softmax on sharded axis triggers AllReduce?
            probs = softmax(scores, axis=-1)
            
            # 5. Context = Probs @ V
            # V: [B, H, S, D] (Same as Q) <dp, *, tp, *>
            # Probs: [B, H, S, S]
            context = ops.matmul(probs, q) 
            
            return context
            
        trace = debug.capture_trace(attention_op, X)
        print(trace)
        
        trace_str = str(trace)
        
        # Verification: Communication Ops MUST exist
        # We expect implicit communication to resolve the conflict.
        # Could be 'reshard', 'all_gather', or 'all_reduce' depending on how it resolved.
        # Most likely 'all_gather' or 'reshard'.
        has_comm = "all_gather" in trace_str or "reshard" in trace_str
        self.assertTrue(has_comm, "Expected communication (all_gather/reshard) to resolve sequence conflict, but found none.")
        
        # Numerical Verification
        import asyncio
        async def verify():
            out_tensor = trace.outputs[0] if isinstance(trace.outputs, (list, tuple)) else trace.outputs
            await out_tensor.realize
            actual = out_tensor.to_numpy()
            
            # Get input values
            await X.realize
            np_x = X.to_numpy()
            
            # NumPy Reference Attention
            def np_softmax(z, axis=-1):
                e = np.exp(z - np.max(z, axis=axis, keepdims=True))
                return e / np.sum(e, axis=axis, keepdims=True)
            
            # 1. Q = swap(x, 1, 2) -> [B, H, S, D]
            q = np.swapaxes(np_x, 1, 2)
            
            # 2. K_T = swap(swap(x, 1, 2), 2, 3) -> [B, H, D, S]
            k_temp = np.swapaxes(np_x, 1, 2)
            k_t = np.swapaxes(k_temp, 2, 3)
            
            # 3. Scores = Q @ K_T -> [B, H, S, S]
            scores = q @ k_t
            
            # 4. Probs = softmax(scores, axis=-1)
            probs = np_softmax(scores, axis=-1)
            
            # 5. Context = Probs @ Q -> [B, H, S, D]
            expected = probs @ q
            
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            print(f"\n✓ Numerical verification PASSED: Shape {actual.shape}")
            
        asyncio.run(verify())
        
        # Also check that we didn't crash :)

if __name__ == "__main__":
    unittest.main()
