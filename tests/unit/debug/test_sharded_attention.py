"""
Sharded Attention Test - GPU Ready

Tests sharding propagation through Multi-Head Attention with Hybrid Sharding (DP+TP).
Auto-detects GPU availability for distributed execution.

For 2-GPU testing: Uses a 1x2 mesh with TP only.
For 4-GPU testing: Uses a 2x2 mesh with DP+TP.
"""

import asyncio
import unittest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DimSpec
from nabla.utils import debug
import nabla.ops as ops
from nabla.ops import view

# Import shared utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "sharding"))
from test_utils import create_mesh, get_mode_string, get_accelerator_count


def softmax(x, axis=-1):
    """Safe softmax for tracing."""
    e = ops.exp(x)
    s = ops.reduce_sum(e, axis=axis, keepdims=True)
    return ops.div(e, s)


class TestShardedAttention(unittest.TestCase):
    
    def test_sharded_mha_2gpu(self):
        """
        2-GPU test: Tensor Parallel attention (no DP).
        Uses 1x2 mesh for testing on 2 GPUs.
        """
        print(f"\n\n{'='*70}")
        print("TRACE: Sharded MHA - 2 GPU (TP Only)")
        print(f"Mode: {get_mode_string()}")
        print('='*70)
        
        # 1x2 mesh: TP only
        mesh = create_mesh("tp_mesh", shape=(2,), axis_names=("tp",))
        
        # Smaller dims for 2-GPU test
        B, S = 2, 4
        Heads, HeadDim = 2, 8
        D_Model = Heads * HeadDim  # 16
        
        # Input: replicated
        X = (Tensor.ones((B, S, D_Model)) * 0.1).shard(mesh, [DimSpec([]), DimSpec([]), DimSpec([])])
        
        # Weights: Column parallel (sharded on output dim)
        W_q = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        W_k = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        W_v = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        W_o = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec(["tp"]), DimSpec([])])

        def mha_forward(x, w_q, w_k, w_v, w_o):
            q_proj = ops.matmul(x, w_q)
            k_proj = ops.matmul(x, w_k)
            v_proj = ops.matmul(x, w_v)
            
            q_view = view.reshape(q_proj, (B, S, Heads, HeadDim))
            k_view = view.reshape(k_proj, (B, S, Heads, HeadDim))
            v_view = view.reshape(v_proj, (B, S, Heads, HeadDim))
            
            q = view.swap_axes(q_view, 1, 2)
            k = view.swap_axes(k_view, 1, 2)
            v = view.swap_axes(v_view, 1, 2)
            
            k_t = view.swap_axes(k, 2, 3)
            scores = ops.matmul(q, k_t)
            probs = softmax(scores, axis=-1)
            context = ops.matmul(probs, v)
            
            context_swap = view.swap_axes(context, 1, 2)
            context_flat = view.reshape(context_swap, (B, S, D_Model))
            
            out = ops.matmul(context_flat, w_o)
            return out
            
        trace = debug.capture_trace(mha_forward, X, W_q, W_k, W_v, W_o)
        print(trace)
        
        # Numerical verification
        out_tensor = trace.outputs[0] if isinstance(trace.outputs, (list, tuple)) else trace.outputs
        asyncio.run(out_tensor.realize)
        actual = out_tensor.to_numpy()
        
        # Numpy reference
        asyncio.run(X.realize)
        asyncio.run(W_q.realize)
        asyncio.run(W_k.realize)
        asyncio.run(W_v.realize)
        asyncio.run(W_o.realize)
        
        np_x = X.to_numpy()
        np_wq = W_q.to_numpy()
        np_wk = W_k.to_numpy()
        np_wv = W_v.to_numpy()
        np_wo = W_o.to_numpy()
        
        def np_softmax(z, axis=-1):
            e = np.exp(z - np.max(z, axis=axis, keepdims=True))
            return e / np.sum(e, axis=axis, keepdims=True)
        
        q_proj = np_x @ np_wq
        k_proj = np_x @ np_wk
        v_proj = np_x @ np_wv
        
        q_view = q_proj.reshape(B, S, Heads, HeadDim)
        k_view = k_proj.reshape(B, S, Heads, HeadDim)
        v_view = v_proj.reshape(B, S, Heads, HeadDim)
        
        q = np.swapaxes(q_view, 1, 2)
        k = np.swapaxes(k_view, 1, 2)
        v = np.swapaxes(v_view, 1, 2)
        
        k_t = np.swapaxes(k, 2, 3)
        scores = q @ k_t
        probs = np_softmax(scores, axis=-1)
        context = probs @ v
        
        context_swap = np.swapaxes(context, 1, 2)
        context_flat = context_swap.reshape(B, S, D_Model)
        expected = context_flat @ np_wo
        
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        print(f"\n✓ 2-GPU MHA numerical verification PASSED: Shape {actual.shape}")

    def test_sharded_mha_4gpu(self):
        """
        4-GPU test: Hybrid DP+TP attention.
        Uses 2x2 mesh. Skips if <4 accelerators (runs simulated on CPU).
        """
        print(f"\n\n{'='*70}")
        print("TRACE: Sharded MHA - 4 GPU (DP+TP)")
        print(f"Mode: {get_mode_string()}")
        print('='*70)
        
        # 2x2 mesh: DP on batch, TP on heads
        mesh = create_mesh("cluster", shape=(2, 2), axis_names=("dp", "tp"))
        
        B, S = 4, 8
        Heads, HeadDim = 4, 16
        D_Model = Heads * HeadDim
        
        X = (Tensor.ones((B, S, D_Model)) * 0.1).shard(mesh, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])
        W_q = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        W_k = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        W_v = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec([]), DimSpec(["tp"])])
        W_o = (Tensor.ones((D_Model, D_Model)) * 0.01).shard(mesh, [DimSpec(["tp"]), DimSpec([])])

        def mha_forward(x, w_q, w_k, w_v, w_o):
            q_proj = ops.matmul(x, w_q)
            k_proj = ops.matmul(x, w_k)
            v_proj = ops.matmul(x, w_v)
            
            q_view = view.reshape(q_proj, (B, S, Heads, HeadDim))
            k_view = view.reshape(k_proj, (B, S, Heads, HeadDim))
            v_view = view.reshape(v_proj, (B, S, Heads, HeadDim))
            
            q = view.swap_axes(q_view, 1, 2)
            k = view.swap_axes(k_view, 1, 2)
            v = view.swap_axes(v_view, 1, 2)
            
            k_t = view.swap_axes(k, 2, 3)
            scores = ops.matmul(q, k_t)
            probs = softmax(scores, axis=-1)
            context = ops.matmul(probs, v)
            
            context_swap = view.swap_axes(context, 1, 2)
            context_flat = view.reshape(context_swap, (B, S, D_Model))
            
            out = ops.matmul(context_flat, w_o)
            return out
            
        trace = debug.capture_trace(mha_forward, X, W_q, W_k, W_v, W_o)
        print(trace)
        
        # Verify trace has expected ops
        trace_str = str(trace)
        self.assertIn("matmul", trace_str)
        self.assertIn("swap_axes", trace_str)
        self.assertIn("reshape", trace_str)
        
        print("\n✓ 4-GPU MHA trace verification PASSED")


if __name__ == "__main__":
    unittest.main()
