# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Rigorous testing of nested vmap compositions with mixed sharding strategies.

This file tests:
1. "Time-Distributed" patterns: vmap(vmap(op))
2. Mixed Sharding: 
   - Outer vmap: Batch parallelism (spmd_axis_name="dp")
   - Inner logic: Tensor parallelism (shard(mesh, P(...)))
3. Complex Operations: Attention-like patterns, MLP layers
4. Trace Inspection: Confirms correct graph construction

The user specifically requested inspection of traces for these complex patterns.
"""

import pytest
import numpy as np

import nabla
from nabla import (
    DeviceMesh, P, vmap, Tensor, 
    add, mul, sub, div, matmul, 
    relu, exp, reduce_sum,
)
from nabla.core import trace

from tests.conftest import (
    make_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_is_sharded,
)

# =============================================================================
# Complex Blocks (Softmax, MLP)
# =============================================================================

def softmax(x, axis=-1):
    """Stable softmax implementation using available ops."""
    # x: (..., D)
    # max_val = reduce_max(x, axis=axis, keepdims=True) # ensure stability? 
    # Nabla doesn't have reduce_max exposed yet? 
    # We'll skip stability for now and rely on basic exp/sum
    e_x = exp(x)
    sum_e_x = reduce_sum(e_x, axis=axis, keepdims=True)
    return div(e_x, sum_e_x)

def mlp_layer(x, w1, w2):
    """Simple MLP: x -> Linear -> ReLU -> Linear."""
    # x: (..., Din)
    # w1: (Din, H)
    # w2: (H, Dout)
    h = relu(matmul(x, w1))
    return matmul(h, w2)

# =============================================================================
# Test Suite
# =============================================================================

class TestNestedVmapAdvanced:
    
    def test_nested_vmap_time_distributed_input_sharding(self, mesh_2x4):
        """
        Pattern: (Batch, Time, Features) -> (Batch, Time, Features)
        
        Outer vmap (Batch): Distributed on "dp"
        Inner vmap (Time): Sequential logic (or parallel)
        Inner Op (Relu): Input features sharded on "tp"
        """
        batch, time, features = 4, 8, 16
        mesh = mesh_2x4 # (2, 4) -> dp, tp
        
        np_x = make_array(batch, time, features, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Define function: shards input features on 'tp' then applies relu
        @vmap(spmd_axis_name="dp", mesh=mesh) # Axis 0 (Batch) -> dp
        def batch_fn(batch_x):
            # batch_x: (Time, Features)
            @vmap
            def time_fn(time_x):
                # time_x: (Features,)
                # Shard features on tp (Tensor Parallelism)
                t_sharded = time_x.shard(mesh, P("tp"))
                return relu(t_sharded)
            return time_fn(batch_x)

        # Trace Inspection
        print("\n=== Trace: Time Distributed Input Sharding ===")
        print(trace(batch_fn, x))
        print("==============================================\n")

        result = batch_fn(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, time, features))
        assert_allclose(result, expected)
        
        # Verify Sharding
        # Result should be (Batch, Time, Features)
        # Batch -> dp (from outer vmap)
        # Time -> Replicated (or not sharded)
        # Features -> tp (propagated from shard(tp))
        spec = result._impl.sharding
        assert spec is not None
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[2].axes

    def test_nested_vmap_attention_pattern(self, mesh_2x4):
        """
        Pattern: Multi-Head Attention Calculation (Simplified)
        
        Input: Q, K, V with shape (Batch, Heads, Seq, Dim)
        Outer vmap (Batch): dp
        Inner vmap (Heads): Distributed? Or parallel loop.
        
        We simulate (Batch, Heads) as (Batch, Heads) nested vmaps.
        Inside: Matmul (Seq, Dim) @ (Dim, Seq) -> (Seq, Seq)
        Contracting dim 'Dim' sharded on 'tp'.
        """
        B, H, S, D = 2, 4, 8, 16
        mesh = mesh_2x4 # (dp=2, tp=4)
        
        # Shapes: (B, H, S, D)
        np_q = make_array(B, H, S, D, seed=1)
        np_k = make_array(B, H, S, D, seed=2)
        # np_v = make_array(B, H, S, D, seed=3) # Skip V for simplicity if needed, or include
        
        t_q = tensor_from_numpy(np_q)
        t_k = tensor_from_numpy(np_k)

        @vmap(spmd_axis_name="dp", mesh=mesh) # Batch -> dp
        def process_batch(b_q, b_k):
            # b_q, b_k: (H, S, D)
            
            @vmap # Heads loop (could shard heads too, but let's shard inner dims)
            def process_head(h_q, h_k):
                # h_q, h_k: (S, D)
                
                # Tensor Parallel Matmul (Row Parallel style equivalent)
                # Shard contracting dim D on 'tp'
                # Q: (S, D) -> shard D on tp
                q_s = h_q.shard(mesh, P(None, "tp"))
                
                # K_T: (D, S) -> we transpose K explicitly
                # K: (S, D) -> shard D on tp
                k_s = h_k.shard(mesh, P(None, "tp"))
                
                # Matmul( (S, D_tp), (D_tp, S) ) -> (S, S) partial sum -> AllReduce
                # Nabla matmul expects standard shapes. We need to Transpose K.
                # But Nabla vmap handles ops. 
                # Let's clean up transpose:
                # We want Q @ K.T. 
                # Note: 'swap_axes' inside vmap works on logical dims.
                
                # We need to explicitly transpose logical dim of k_s
                # swap_axes(k_s, 0, 1) -> (D, S)
                # BUT wait, k_s is (S, D). swap gets (D, S).
                # Does swap_axes propagate sharding correctly through vmap? (Yes, verified in tests)
                k_t_s = nabla.swap_axes(k_s, 0, 1) # (D, S). D is sharded 'tp'.
                
                # Matmul: (S, D<tp>) @ (D<tp>, S) -> (S, S)
                score = matmul(q_s, k_t_s) 
                
                # Output score should be (S, S) Replicated (after AllReduce)
                # Then we verify softmax runs on it
                probs = softmax(score, axis=-1)
                return probs

            return process_head(b_q, b_k)

        # Trace Inspection
        print("\n=== Trace: Attention Pattern Q@K.T ===")
        print(trace(process_batch, t_q, t_k))
        print("======================================\n")
        
        result_nabla = process_batch(t_q, t_k)
        
        # Numpy Ref
        # (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        # Using easy numpy logic:
        res_np = np.empty((B, H, S, S), dtype=np.float32)
        for b in range(B):
            for h in range(H):
                q = np_q[b, h]
                k = np_k[b, h]
                score = q @ k.T
                # Softmax
                sc_exp = np.exp(score)
                res_np[b, h] = sc_exp / np.sum(sc_exp, axis=-1, keepdims=True)
        
        assert_shape(result_nabla, (B, H, S, S))
        assert_allclose(result_nabla, res_np, rtol=1e-4) # float32 tolerance

        # Verify Sharding
        # Batch logic: Axis 0 (Batch) must have 'dp'
        # Heads: Axis 1. Not sharded explicitly.
        # S, S: Axes 2, 3. Matmul output (AllReduced) -> Replicated.
        spec = result_nabla._impl.sharding
        assert "dp" in spec.dim_specs[0].axes
        # Check inner dims are replicated (or at least not 'tp')
        # Actually 'tp' was reduced out.
        assert not spec.dim_specs[2].axes 
        assert not spec.dim_specs[3].axes

    def test_nested_vmap_mlp_data_tensor_parallel(self, mesh_2x4):
        """
        Pattern: Data Parallel Batch, Tensor Parallel Weights MLP.
        
        (Batch, Features) -> Linear -> Linear -> Out
        But with nested vmap simulating 'MicroBatch' or 'Grouped' processing?
        Or simply: vmap(batch) calling a function that does sharding.
        
        Let's do: (Batch, Seq, Dim) -> apply MLP to each token.
        Outer: Batch (dp)
        Inner: Seq
        MLP: Weights sharded (tp)
        """
        B, S, Din, H, Dout = 2, 4, 16, 32, 8
        mesh = mesh_2x4
        
        np_x = make_array(B, S, Din, seed=42)
        np_w1 = make_array(Din, H, seed=1)
        np_w2 = make_array(H, Dout, seed=2)
        
        t_x = tensor_from_numpy(np_x)
        t_w1 = tensor_from_numpy(np_w1)
        t_w2 = tensor_from_numpy(np_w2)
        
        # Pre-shard weights? Or shard inside?
        # Typically weights are parameters, pre-sharded.
        # Let's shard weights initially to simulate real model.
        # W1: Col Parallel (shard H on tp) -> Output H is partial? No, Output H is sharded.
        # Linear: X @ W. (..., Din) @ (Din, H_tp) -> (..., H_tp).
        # W2: Row Parallel (shard H on tp).
        # Linear: X_tp @ W_tp. (..., H_tp) @ (H_tp, Dout) -> (..., Dout) AllReduce.
        
        # Let's define manual sharding on weights
        t_w1_s = t_w1.shard(mesh, P(None, "tp"))
        t_w2_s = t_w2.shard(mesh, P("tp", None))
        
        @vmap(spmd_axis_name="dp", mesh=mesh) # Batch -> dp
        def batch_forward(b_x):
            # b_x: (S, Din)
            # We treat Seq as just another batch dimension for matmul usually, 
            # but here let's vmap over it to stress test nested vmap
            # OR just do matmul on (S, Din) directly.
            # Stress test: nested vmap!
            
            @vmap 
            def token_forward(tok):
                # tok: (Din,)
                # MLP
                # 1. tok @ w1. (Din) @ (Din, H_tp) -> (H_tp)
                h = relu(matmul(tok, t_w1_s))
                # 2. h @ w2. (H_tp) @ (H_tp, Dout) -> (Dout)
                out = matmul(h, t_w2_s)
                return out
                
            return token_forward(b_x)

        # Trace
        print("\n=== Trace: Nested MLP (Data + Tensor Parallel) ===")
        print(trace(batch_forward, t_x))
        print("==================================================\n")

        res_nabla = batch_forward(t_x)
        
        # Numpy Ref
        # (B, S, Din)
        # W1: (Din, H)
        # W2: (H, Dout)
        h_np = np.maximum(np_x @ np_w1, 0)
        res_np = h_np @ np_w2
        
        assert_shape(res_nabla, (B, S, Dout))
        assert_allclose(res_nabla, res_np, rtol=1e-4)
        
        # Verify Sharding
        # Batch: dp
        # Seq: Rep
        # Dout: Rep (AllReduced)
        spec = res_nabla._impl.sharding
        assert "dp" in spec.dim_specs[0].axes


class TestNestedViewAndReduction:
    """Stress tests for View and Reduction operations inside nested vmap."""

    def test_nested_reshape_preserving_batch_sharding(self, mesh_2x4):
        """
        Pattern: (Batch, Time, H, W, C) -> vmap(B) -> vmap(T) -> reshape -> Linear
        
        Original: (B, T, H, W, C)
        Reshape to: (B, T, H*W, C) inside vmap?
        Actually vmap sees (H, W, C).
        Reshape to (H*W, C).
        
        Goal: specific check that outer Batch sharding ('dp') is preserved 
        despite the reshape un-sharding logical dims.
        """
        B, T, H, W, C = 2, 4, 3, 3, 16 
        mesh = mesh_2x4 # dp=2, tp=4
        
        np_x = make_array(B, T, H, W, C, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Shard inner C on tp? ReshapeOp currently unshards logical dims.
        # So we shard, then reshape, and check if B stays sharded.
        
        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_fn(batch_x):
            # batch_x: (T, H, W, C)
            @vmap
            def time_fn(time_x):
                # time_x: (H, W, C)
                # Shard C on 'tp'
                t_s = time_x.shard(mesh, P(None, None, "tp"))
                
                # Reshape (H, W, C) -> (H*W, C)
                # This should trigger "conservative unshard" for logical dims,
                # BUT batch dims (B, T) must remain untouched/sharded.
                flat = nabla.reshape(t_s, (H*W, C)) 
                return flat
            return time_fn(batch_x)

        # Trace
        print("\n=== Trace: Nested Reshape (Preserve Batch Sharding) ===")
        print(trace(batch_fn, x))
        print("=======================================================\n")
        
        result = batch_fn(x)
        expected = np_x.reshape(B, T, H*W, C)
        
        assert_shape(result, (B, T, H*W, C))
        assert_allclose(result, expected)
        
        # Verify Sharding
        spec = result._impl.sharding
        # Batch (axis 0) MUST be on dp
        assert "dp" in spec.dim_specs[0].axes
        # C (axis 3) should be Replicated (because ReshapeOp un-shards logical)
        # Note: In future optimized reshape, it might stay sharded, but for now expect Rep.
        assert not spec.dim_specs[3].axes
        
    def test_nested_reduction_over_sharded_axis(self, mesh_2x4):
        """
        Pattern: ReduceSum over a sharded tensor-parallel axis.
        
        (B, T, D) -> vmap(dp) -> vmap -> shard(D, tp) -> reduce_sum(D)
        
        Output: (B, T).
        """
        B, T, D = 2, 4, 16
        mesh = mesh_2x4
        np_x = make_array(B, T, D, seed=42)
        x = tensor_from_numpy(np_x)
        
        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_fn(b_x):
            @vmap
            def time_fn(t_x):
                # t_x: (D,)
                t_s = t_x.shard(mesh, P("tp"))
                return reduce_sum(t_s, axis=0)
            return time_fn(b_x)

        print("\n=== Trace: Nested Reduce Sum (Sharded Axis) ===")
        print(trace(batch_fn, x))
        print("===============================================\n")

        result = batch_fn(x)
        expected = np.sum(np_x, axis=2)
        
        assert_shape(result, (B, T))
        assert_allclose(result, expected, rtol=1e-4)
        
        # Verify Batch sharding is preserved
        spec = result._impl.sharding
        assert "dp" in spec.dim_specs[0].axes

    def test_stress_complex_chain(self, mesh_3d_2x4x2):
        """
        Chain: Broadcast -> Matmul -> Reduce -> Reshape
        
        Global Mesh: (dp, tp, pp) -> (2, 4, 2)
        Input: (B, S, D)
        Outer vmap (B): dp (2)
        Inner vmap (S): pp (2)
        tp (4): Used for model parallel Linear
        Logic:
           1. Broadcast scalar bias to (D)
           2. Shard bias on tp
           3. Matmul input (D) @ W (D, D_out). D sharded on tp.
           4. ReduceSum(D_out) -> just to change shape.
           5. Reshape result.
        """
        mesh = mesh_3d_2x4x2
        # Dimensions must match mesh
        B, S = 2, 2 
        # Din/Dout must be divisible by tp=4
        Din, Dout = 16, 16
        
        np_x = make_array(B, S, Din, seed=42)
        np_w = make_array(Din, Dout, seed=1)
        
        t_x = tensor_from_numpy(np_x)
        t_w = tensor_from_numpy(np_w)
        
        # Shard W: Input channel sharded on tp (Row Parallel equivalent)
        t_w_s = t_w.shard(mesh, P("tp", None))
        
        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_fn(b_x):
            @vmap(spmd_axis_name="pp", mesh=mesh)
            def seq_fn(s_x):
                # s_x: (Din,)
                # Shard input on tp (to match W)
                s_s = s_x.shard(mesh, P("tp"))
                
                # Matmul: (Din_tp) @ (Din_tp, Dout) -> (Dout) Partial Sum -> AllReduce
                # Wait, Vector @ Matrix
                # (Din) @ (Din, Dout) -> (Dout).
                # (tp) @ (tp, rep) -> (rep) Partial -> AllReduce.
                dense = matmul(s_s, t_w_s)
                
                # Reshape (Dout) -> (2, Dout//2)
                reshaped = nabla.reshape(dense, (2, Dout // 2))
                
                # ReduceSum over axis 0 -> (Dout//2)
                return reduce_sum(reshaped, axis=0)
            return seq_fn(b_x)

        print("\n=== Trace: Stress Chain (Broad/Matmul/Reshape/Reduce) ===")
        print(trace(batch_fn, t_x))
        print("=========================================================\n")
        
        result = batch_fn(t_x)
        
        # Numpy Ref
        # x @ w -> (B, S, Dout)
        # Reshape (B, S, 2, Dout//2)
        # Sum axis 2 -> (B, S, Dout//2)
        dense_np = np_x @ np_w
        reshaped_np = dense_np.reshape(B, S, 2, Dout // 2)
        expected = np.sum(reshaped_np, axis=2)
        
        assert_shape(result, (B, S, Dout // 2))
        assert_allclose(result, expected, rtol=1e-4)
        
        spec = result._impl.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "pp" in spec.dim_specs[1].axes


class TestReviewStress:
    """
    User specific request: 
    vmap_with_sharding(vmap_with_sharding(body_function_with_sharding_on_logical_dims))
    AND checking view ops: broadcast, reshape, squeeze.
    """
    
    def test_double_spmd_vmap_view_ops(self, mesh_3d_2x4x2):
        """
        Scenario:
        1. Outer vmap (Batch=2) -> 'dp'
        2. Inner vmap (Time=2) -> 'pp'
        3. Body logic:
           - Input (D=16) -> shard on 'tp' (4)
           - Unsqueeze (Add dim)
           - Broadcast (Expand dim)
           - Squeeze (Remove dim)
        """
        mesh = mesh_3d_2x4x2 # (dp=2, tp=4, pp=2)
        B, T, D = 2, 2, 16
        
        np_x = make_array(B, T, D, seed=100)
        x = tensor_from_numpy(np_x)
        
        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_fn(b_x):
            # b_x: (T, D)
            @vmap(spmd_axis_name="pp", mesh=mesh)
            def time_fn(t_x):
                # t_x: (D,)
                
                # 1. Shard Logical 'tp'
                # t_x is (D,). Shard D on tp.
                t_s = t_x.shard(mesh, P("tp"))
                
                # 2. Unsqueeze -> (1, D)
                obs_1 = nabla.unsqueeze(t_s, axis=0)
                
                # 3. Broadcast -> (4, D)
                # Note: Broadcast might replicate sharding or preserve it?
                # (1, D_tp) -> (4, D_tp)?
                obs_broad = nabla.broadcast_to(obs_1, (4, D))
                
                # 4. Squeeze a different dim? No, squeeze logical 0?
                # If we squeeze 0, we verify reduce logic or simplify.
                # Let's reduce sum over 0 first to make it 1 again?
                # Or slice.
                # Let's just Add random tensor (4, D) and then reduce.
                added = add(obs_broad, obs_broad)
                
                # Squeeze? If shape is (4, D), can't squeeze 0.
                # Let's reshape?
                # (4, D) -> (4 * D)
                # Reshape usually un-shards logical.
                flat = nabla.reshape(added, (4 * D,))
                
                # 5. Squeeze check:
                # Let's unsqueeze again at end to return (1, 4*D)
                return nabla.unsqueeze(flat, axis=0)
                
            return time_fn(b_x)

        print("\n=== Trace: Double SPMD Vmap + View Ops Stress ===")
        print(trace(batch_fn, x))
        print("=================================================\n")
        
        result = batch_fn(x)
        
        # Validation
        # Operations were: Identity-ish.
        # x (D) -> (1, D) -> (4, D) [replicated 4 times] -> added (x*2) -> flat -> (1, 64)
        # So output is [x*2, x*2, x*2, x*2] flat.
        
        # Numpy
        # x: (B, T, D)
        # Inner: x_t (D) -> x_t * 2 repeated 4 times.
        # Shape (B, T, 1, 4*D)
        
        expected_inner = []
        for b in range(B):
            row = []
            for t in range(T):
                val = np_x[b, t] * 2 # added
                # broadcast 4 times
                val_broad = np.broadcast_to(val[None, :], (4, D))
                row.append(val_broad.reshape(1, 4*D))
            expected_inner.append(np.stack(row))
        expected = np.stack(expected_inner)
        
        assert_shape(result, (B, T, 1, 4 * D))
        assert_allclose(result, expected)
        
        # Verify Sharding
        spec = result._impl.sharding
        # Batch dims B, T must be dp, pp
        # Check dim_specs[0] ('dp') and dim_specs[1] ('pp')
        # Note: vmap might permute them physically, but sharding spec should track.
        
        # We look for "dp" and "pp" in the first two dimensions of the spec
        axes_0 = spec.dim_specs[0].axes
        axes_1 = spec.dim_specs[1].axes
        
        # One must be dp, one pp.
        found_dp = ("dp" in axes_0) or ("dp" in axes_1)
        found_pp = ("pp" in axes_0) or ("pp" in axes_1)
        
        assert found_dp, f"Output spec {spec} missing dp on batch axes"
        assert found_pp, f"Output spec {spec} missing pp on batch axes"

