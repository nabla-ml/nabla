
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
    # Safe softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    # Note: nabla doesn't have reduce_max yet, so we use simple exp(x) / sum(exp(x))
    # robust enough for tracing.
    e = ops.exp(x)
    s = ops.reduce_sum(e, axis=axis, keepdims=True)
    return ops.div(e, s)

class TestShardedAttention(unittest.TestCase):
    
    def test_sharded_mha_trace(self):
        """
        Test sharding propagation through a Multi-Head Attention block.
        Target: Hybrid Sharding (Batch on DP, Heads on TP).
        """
        print("\n\n================================================================================")
        print("TRACE: Distributed Multi-Head Attention (DP+TP)")
        print("================================================================================")
        
        # 1. Setup Mesh (2, 2)
        dp, tp = 2, 2
        mesh = get_mesh((dp, tp), ("dp", "tp"))
        
        # 2. Dimensions
        B, S = 4, 8
        Heads, HeadDim = 4, 16
        D_Model = Heads * HeadDim # 64
        
        # 3. Tensors
        # Input: [Batch, Seq, D_Model] sharded [dp, *, *]
        # Use ones for numerical stability
        X = Tensor.ones((B, S, D_Model)).shard(mesh, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])
        
        # Weights: [D_Model, D_Model] sharded [*, tp] (Col Parallel)
        # Use small values to prevent softmax overflow
        W_q = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        W_k = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        W_v = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        
        # Output Projection: Row Parallel [tp, *] (Contracting dim split)
        W_o = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec(["tp"]), DimSpec([])])

        def mha_forward(x, w_q, w_k, w_v, w_o):
            # 1. Projections
            # x: [B, S, D] <dp, *>
            # w: [D, D] <*, tp>
            # q: [B, S, D] <dp, *, tp> (Heads split)
            q_proj = ops.matmul(x, w_q)
            k_proj = ops.matmul(x, w_k)
            v_proj = ops.matmul(x, w_v)
            
            # 2. Split Heads (Reshape + Swap)
            # [B, S, D] -> [B, S, H, D]
            # Sharding <dp, *, tp> -> <dp, *, tp, *> ? No.
            # D (64) -> H(4) * D(16).
            # If D is sharded by tp(2), then H is sharded by tp(2) and D(16) is local.
            # Target: [B, S, H, D] <dp, *, tp, *>
            q_view = view.reshape(q_proj, (B, S, Heads, HeadDim))
            k_view = view.reshape(k_proj, (B, S, Heads, HeadDim))
            v_view = view.reshape(v_proj, (B, S, Heads, HeadDim))
            
            # Swap to [B, H, S, D] <dp, tp, *, *>
            q = view.swap_axes(q_view, 1, 2)
            k = view.swap_axes(k_view, 1, 2)
            v = view.swap_axes(v_view, 1, 2)
            
            # 3. Attention Scores
            # Q @ K.T
            # Q: [B, H, S, D]
            # K_T: [B, H, D, S] (Swap last two)
            k_t = view.swap_axes(k, 2, 3)
            # Matmul: Batch dims (B, H) broadcast/match.
            # H is sharded <tp>. Matmul should support sharded batch dims.
            scores = ops.matmul(q, k_t) # [B, H, S, S] <dp, tp, *, *>
            
            # Scale
            scale = 1.0 / np.sqrt(HeadDim)
            # scores = scores * scale (Using mul op not implemented as scalar mul yet?)
            # Simulate scalar mul with broadcast
            # scores = ops.mul(scores, Tensor(scale)) # Skip for trace simplicity
            
            # 4. Softmax
            # Local reduction on axis -1 (S).
            probs = softmax(scores, axis=-1)
            
            # 5. Context
            # Probs @ V
            # [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
            context = ops.matmul(probs, v) # <dp, tp, *, *>
            
            # 6. Merge Heads
            # [B, H, S, D] -> [B, S, H, D]
            context_swap = view.swap_axes(context, 1, 2)
            # [B, S, H*D] <dp, *, tp>
            context_flat = view.reshape(context_swap, (B, S, D_Model))
            
            # 7. Output Projection
            # [B, S, D] <dp, *, tp> @ [D, D] <tp, *>
            # Contracting dim D is sharded <tp>.
            # Should auto-insert AllReduce.
            out = ops.matmul(context_flat, w_o) # <dp, *, *>
            
            return out
            
        trace = debug.capture_trace(mha_forward, X, W_q, W_k, W_v, W_o)
        print(trace)
        
        # Verify Final Output Sharding
        # Expect [dp, *, *] -> [{"dp"}, {}, {}]
        # Verify Final Output Sharding
        # Expect [dp, *, *] -> [{"dp"}, {}, {}]
        out_tensor = trace.outputs[0] if isinstance(trace.outputs, (list, tuple)) else trace.outputs
        out_spec = out_tensor.sharding
        self.assertEqual(len(out_spec.dim_specs), 3)
        self.assertEqual(out_spec.dim_specs[0].axes, ["dp"])
        self.assertEqual(out_spec.dim_specs[1].axes, [])
        self.assertEqual(out_spec.dim_specs[2].axes, [])
        
        # Verify Intermediate Sharding (Heads on TP)
        # We need to inspect nodes in the trace.
        # This is harder programmatically without walking the graph.
        # But GraphPrinter output will show it for visual confirm.
        
        # Check for AllReduce (from Step 7 output projection)
        trace_str = str(trace)
        self.assertIn("all_reduce", trace_str)
        self.assertIn("swap_axes", trace_str)
        self.assertIn("reshape", trace_str)
        
        # Numerical Verification
        import asyncio
        async def verify():
            out_tensor_v = trace.outputs[0] if isinstance(trace.outputs, (list, tuple)) else trace.outputs
            await out_tensor_v.realize
            actual = out_tensor_v.to_numpy()
            
            # Get input values
            await X.realize
            await W_q.realize
            await W_k.realize
            await W_v.realize
            await W_o.realize
            
            np_x = X.to_numpy()
            np_wq = W_q.to_numpy()
            np_wk = W_k.to_numpy()
            np_wv = W_v.to_numpy()
            np_wo = W_o.to_numpy()
            
            # NumPy Reference MHA
            def np_softmax(z, axis=-1):
                e = np.exp(z - np.max(z, axis=axis, keepdims=True))
                return e / np.sum(e, axis=axis, keepdims=True)
            
            # 1. Projections
            q_proj = np_x @ np_wq
            k_proj = np_x @ np_wk
            v_proj = np_x @ np_wv
            
            # 2. Split Heads
            q_view = q_proj.reshape(B, S, Heads, HeadDim)
            k_view = k_proj.reshape(B, S, Heads, HeadDim)
            v_view = v_proj.reshape(B, S, Heads, HeadDim)
            
            # Swap to [B, H, S, D]
            q = np.swapaxes(q_view, 1, 2)
            k = np.swapaxes(k_view, 1, 2)
            v = np.swapaxes(v_view, 1, 2)
            
            # 3. Attention Scores
            k_t = np.swapaxes(k, 2, 3)
            scores = q @ k_t # [B, H, S, S]
            
            # 4. Softmax
            probs = np_softmax(scores, axis=-1)
            
            # 5. Context
            context = probs @ v # [B, H, S, D]
            
            # 6. Merge Heads
            context_swap = np.swapaxes(context, 1, 2)
            context_flat = context_swap.reshape(B, S, D_Model)
            
            # 7. Output Projection
            expected = context_flat @ np_wo
            
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            print(f"\n✓ Numerical verification PASSED: Shape {actual.shape}")
            
        asyncio.run(verify())
        
if __name__ == "__main__":
    unittest.main()
