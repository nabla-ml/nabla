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

import numpy as np

import nabla
from nabla import (
    P,
    add,
    div,
    exp,
    matmul,
    reduce_sum,
    relu,
    vmap,
)
from nabla.core import trace
from tests.conftest import (
    assert_allclose,
    assert_shape,
    make_array,
    tensor_from_numpy,
)


def softmax(x, axis=-1):
    """Stable softmax implementation using available ops."""

    e_x = exp(x)
    sum_e_x = reduce_sum(e_x, axis=axis, keepdims=True)
    return div(e_x, sum_e_x)


def mlp_layer(x, w1, w2):
    """Simple MLP: x -> Linear -> ReLU -> Linear."""

    h = relu(matmul(x, w1))
    return matmul(h, w2)


class TestNestedVmapAdvanced:

    def test_nested_vmap_time_distributed_input_sharding(self, mesh_2x4):
        """
        Pattern: (Batch, Time, Features) -> (Batch, Time, Features)

        Outer vmap (Batch): Distributed on "dp"
        Inner vmap (Time): Sequential logic (or parallel)
        Inner Op (Relu): Input features sharded on "tp"
        """
        batch, time, features = 4, 8, 16
        mesh = mesh_2x4

        np_x = make_array(batch, time, features, seed=42)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_fn(batch_x):

            @vmap
            def time_fn(time_x):

                t_sharded = time_x.shard(mesh, P("tp"))
                return relu(t_sharded)

            return time_fn(batch_x)

        print("\n=== Trace: Time Distributed Input Sharding ===")
        print(trace(batch_fn, x))
        print("==============================================\n")

        result = batch_fn(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (batch, time, features))
        assert_allclose(result, expected)

        spec = result.sharding
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
        mesh = mesh_2x4

        np_q = make_array(B, H, S, D, seed=1)
        np_k = make_array(B, H, S, D, seed=2)

        t_q = tensor_from_numpy(np_q)
        t_k = tensor_from_numpy(np_k)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def process_batch(b_q, b_k):

            @vmap
            def process_head(h_q, h_k):

                q_s = h_q.shard(mesh, P(None, "tp"))

                k_s = h_k.shard(mesh, P(None, "tp"))

                k_t_s = nabla.swap_axes(k_s, 0, 1)

                score = matmul(q_s, k_t_s)

                probs = softmax(score, axis=-1)
                return probs

            return process_head(b_q, b_k)

        print("\n=== Trace: Attention Pattern Q@K.T ===")
        print(trace(process_batch, t_q, t_k))
        print("======================================\n")

        result_nabla = process_batch(t_q, t_k)

        res_np = np.empty((B, H, S, S), dtype=np.float32)
        for b in range(B):
            for h in range(H):
                q = np_q[b, h]
                k = np_k[b, h]
                score = q @ k.T

                sc_exp = np.exp(score)
                res_np[b, h] = sc_exp / np.sum(sc_exp, axis=-1, keepdims=True)

        assert_shape(result_nabla, (B, H, S, S))
        assert_allclose(result_nabla, res_np, rtol=1e-4)

        spec = result_nabla.sharding
        assert "dp" in spec.dim_specs[0].axes

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

        t_w1_s = t_w1.shard(mesh, P(None, "tp"))
        t_w2_s = t_w2.shard(mesh, P("tp", None))

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_forward(b_x):

            @vmap
            def token_forward(tok):

                h = relu(matmul(tok, t_w1_s))

                out = matmul(h, t_w2_s)
                return out

            return token_forward(b_x)

        print("\n=== Trace: Nested MLP (Data + Tensor Parallel) ===")
        print(trace(batch_forward, t_x))
        print("==================================================\n")

        res_nabla = batch_forward(t_x)

        h_np = np.maximum(np_x @ np_w1, 0)
        res_np = h_np @ np_w2

        assert_shape(res_nabla, (B, S, Dout))
        assert_allclose(res_nabla, res_np, rtol=1e-4)

        spec = res_nabla.sharding
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
        mesh = mesh_2x4

        np_x = make_array(B, T, H, W, C, seed=42)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_fn(batch_x):

            @vmap
            def time_fn(time_x):

                t_s = time_x.shard(mesh, P(None, None, "tp"))

                flat = nabla.reshape(t_s, (H * W, C))
                return flat

            return time_fn(batch_x)

        print("\n=== Trace: Nested Reshape (Preserve Batch Sharding) ===")
        print(trace(batch_fn, x))
        print("=======================================================\n")

        result = batch_fn(x)
        expected = np_x.reshape(B, T, H * W, C)

        assert_shape(result, (B, T, H * W, C))
        assert_allclose(result, expected)

        spec = result.sharding

        assert "dp" in spec.dim_specs[0].axes

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

        spec = result.sharding
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

        B, S = 2, 2

        Din, Dout = 16, 16

        np_x = make_array(B, S, Din, seed=42)
        np_w = make_array(Din, Dout, seed=1)

        t_x = tensor_from_numpy(np_x)
        t_w = tensor_from_numpy(np_w)

        t_w_s = t_w.shard(mesh, P("tp", None))

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_fn(b_x):
            @vmap(spmd_axis_name="pp", mesh=mesh)
            def seq_fn(s_x):

                s_s = s_x.shard(mesh, P("tp"))

                dense = matmul(s_s, t_w_s)

                reshaped = nabla.reshape(dense, (2, Dout // 2))

                return reduce_sum(reshaped, axis=0)

            return seq_fn(b_x)

        print("\n=== Trace: Stress Chain (Broad/Matmul/Reshape/Reduce) ===")
        print(trace(batch_fn, t_x))
        print("=========================================================\n")

        result = batch_fn(t_x)

        dense_np = np_x @ np_w
        reshaped_np = dense_np.reshape(B, S, 2, Dout // 2)
        expected = np.sum(reshaped_np, axis=2)

        assert_shape(result, (B, S, Dout // 2))
        assert_allclose(result, expected, rtol=1e-4)

        spec = result.sharding
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
        mesh = mesh_3d_2x4x2
        B, T, D = 2, 2, 16

        np_x = make_array(B, T, D, seed=100)
        x = tensor_from_numpy(np_x)

        @vmap(spmd_axis_name="dp", mesh=mesh)
        def batch_fn(b_x):

            @vmap(spmd_axis_name="pp", mesh=mesh)
            def time_fn(t_x):

                t_s = t_x.shard(mesh, P("tp"))

                obs_1 = nabla.unsqueeze(t_s, axis=0)

                obs_broad = nabla.broadcast_to(obs_1, (4, D))

                added = add(obs_broad, obs_broad)

                flat = nabla.reshape(added, (4 * D,))

                return nabla.unsqueeze(flat, axis=0)

            return time_fn(b_x)

        print("\n=== Trace: Double SPMD Vmap + View Ops Stress ===")
        print(trace(batch_fn, x))
        print("=================================================\n")

        result = batch_fn(x)

        expected_inner = []
        for b in range(B):
            row = []
            for t in range(T):
                val = np_x[b, t] * 2

                val_broad = np.broadcast_to(val[None, :], (4, D))
                row.append(val_broad.reshape(1, 4 * D))
            expected_inner.append(np.stack(row))
        expected = np.stack(expected_inner)

        assert_shape(result, (B, T, 1, 4 * D))
        assert_allclose(result, expected)

        spec = result.sharding

        axes_0 = spec.dim_specs[0].axes
        axes_1 = spec.dim_specs[1].axes

        found_dp = ("dp" in axes_0) or ("dp" in axes_1)
        found_pp = ("pp" in axes_0) or ("pp" in axes_1)

        assert found_dp, f"Output spec {spec} missing dp on batch axes"
        assert found_pp, f"Output spec {spec} missing pp on batch axes"
