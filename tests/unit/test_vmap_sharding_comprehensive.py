# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comprehensive vmap + sharding tests with realistic patterns.

This file tests the realistic use case of vmap with sharding where:
1. Sharding is applied INSIDE the vmapped function on LOGICAL shapes
2. User thinks in terms of logical dimensions (batch_dims is implicit)
3. Uses JAX-style P(...) syntax for cleaner specs
4. Tests hybrid parallelism patterns (data + tensor parallelism)

Test pattern:
- Create unsharded batched input with shape (batch, *logical_shape)
- Define function f(x) that operates on logical_shape
- INSIDE f, apply sharding using P(...) on logical dimensions
- Apply vmap(f, in_axes=...) with appropriate axis specifications
- Verify numerical correctness and proper sharding propagation

Realistic scenarios tested:
- MLP layers with column/row parallel weights (Megatron-style)
- Data parallelism on batch dimension
- Tensor parallelism on hidden dimensions
- Mixed sharding strategies
"""

import numpy as np
import pytest

from nabla import (
    DeviceMesh,
    P,
    add,
    broadcast_to,
    div,
    exp,
    matmul,
    mean,
    mul,
    neg,
    reduce_sum,
    relu,
    reshape,
    sigmoid,
    squeeze,
    sub,
    swap_axes,
    tanh,
    unsqueeze,
    vmap,
)
from nabla.core import trace
from nabla.ops.multi_output import chunk, split
from .conftest import (
    assert_allclose,
    assert_is_sharded,
    assert_shape,
    make_array,
    make_positive_array,
    tensor_from_numpy,
)


class TestVmapWithShardingUnaryOps:
    """Test vmap with sharding applied INSIDE function on logical shapes."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_vmap_relu_with_sharding_inside(self, mesh_shape, mesh_axes):
        """vmap(relu) where sharding is applied INSIDE on logical shape."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):

            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return relu(x_sharded)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (batch, 8, 16))
        assert_allclose(result, expected)
        assert_is_sharded(result, True)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_vmap_sigmoid_with_feature_sharding(self, mesh_shape, mesh_axes):
        """vmap(sigmoid) with last dimension sharded (feature parallelism)."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(mesh_axes[-1]))
            return sigmoid(x_sharded)

        np_x = make_array(batch, 16, seed=42) * 0.1
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = 1 / (1 + np.exp(-np_x))

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected, rtol=1e-4)


class TestVmapBinaryOpsWithSharding:
    """Test vmap on binary ops with sharding - weights broadcast, data batched."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_vmap_add_batched_data_sharded_bias(self, mesh_shape, mesh_axes):
        """vmap(add) with batched input and broadcast sharded bias (realistic pattern)."""
        batch = 4
        hidden = 16
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x, bias):

            bias_sharded = bias.shard(mesh, P(mesh_axes[-1]))

            return add(x, bias_sharded)

        np_x = make_array(batch, hidden, seed=42)
        np_bias = make_array(hidden, seed=43)

        x = tensor_from_numpy(np_x)
        bias = tensor_from_numpy(np_bias)

        result = vmap(f, in_axes=(0, None))(x, bias)
        expected = np_x + np_bias

        assert_shape(result, (batch, hidden))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_vmap_mul_elementwise_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(mul) with both inputs batched and sharded identically."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x, y):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            y_sharded = y.shard(mesh, P(None, mesh_axes[-1]))
            return mul(x_sharded, y_sharded)

        np_a = make_array(batch, 8, 16, seed=42)
        np_b = make_array(batch, 8, 16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        result = vmap(f)(a, b)
        expected = np_a * np_b

        assert_shape(result, (batch, 8, 16))
        assert_allclose(result, expected)


class TestVmapMatmulSharding:
    """Test vmap(matmul) with tensor parallelism patterns."""

    def test_vmap_matmul_column_parallel(self):
        """vmap(matmul) with column-parallel weights (Megatron-style)."""
        batch = 4
        in_features = 16
        out_features = 32
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        def f(x, w):

            w_sharded = w.shard(mesh, P(None, "tp"))
            result = matmul(x, w_sharded)

            return result

        np_x = make_array(batch, in_features, seed=42)
        np_w = make_array(in_features, out_features, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        result = vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w

        assert_shape(result, (batch, out_features))
        assert_allclose(result, expected, rtol=1e-4)

    def test_vmap_matmul_row_parallel(self):
        """vmap(matmul) with row-parallel weights (Megatron-style with reduction)."""
        batch = 4
        in_features = 32
        out_features = 16
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        def f(x, w):

            x_sharded = x.shard(mesh, P("tp"))

            w_sharded = w.shard(mesh, P("tp", None))
            result = matmul(x_sharded, w_sharded)

            return result

        np_x = make_array(batch, in_features, seed=42)
        np_w = make_array(in_features, out_features, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        result = vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w

        assert_shape(result, (batch, out_features))
        assert_allclose(result, expected, rtol=1e-4)

    def test_vmap_matmul_2d_mesh_hybrid_parallel(self):
        """vmap(matmul) on 2D mesh with data + tensor parallelism."""
        batch = 4
        in_features = 16
        out_features = 32
        mesh = DeviceMesh("mesh", (2, 2), ("dp", "tp"))

        def f(x, w):

            w_sharded = w.shard(mesh, P(None, "tp"))
            result = matmul(x, w_sharded)
            return result

        np_x = make_array(batch, in_features, seed=42)
        np_w = make_array(in_features, out_features, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        result = vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w

        assert_shape(result, (batch, out_features))
        assert_allclose(result, expected, rtol=1e-4)


class TestVmapReductionsWithSharding:
    """Test vmap on reduction ops with sharding."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_vmap_reduce_sum_sharded_input(self, mesh_shape, mesh_axes):
        """vmap(reduce_sum) with sharded input."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))

            return reduce_sum(x_sharded, axis=1)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.sum(np_x, axis=2)

        assert_shape(result, (batch, 8))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
        ],
    )
    def test_vmap_mean_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(mean) with sharded dimension."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(mesh_axes[-1]))
            return mean(x_sharded, axis=0)

        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.mean(np_x, axis=1)

        assert_shape(result, (batch,))
        assert_allclose(result, expected)


class TestVmapViewOpsWithSharding:
    """Test vmap on view ops with sharding."""

    def test_vmap_reshape_merge_sharded_dim(self):
        """vmap(reshape) that merges a sharded dimension.

        This tests the case where reshape merges a sharded dimension with another.
        The library should automatically gather before reshape to maintain correctness.
        """
        batch = 4
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        def f(x):
            x_sharded = x.shard(mesh, P(None, "tp"))
            return reshape(x_sharded, (128,))

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np_x.reshape(batch, 128)

        assert_shape(result, (batch, 128))
        assert_allclose(result, expected)

    def test_vmap_reshape_split_sharded_dim(self):
        """vmap(reshape) that splits a sharded dimension - no gather needed."""
        batch = 4
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        def f(x):
            x_sharded = x.shard(mesh, P("tp"))

            return reshape(x_sharded, (8, 16))

        np_x = make_array(batch, 128, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np_x.reshape(batch, 8, 16)

        assert_shape(result, (batch, 8, 16))
        assert_allclose(result, expected)

    def test_vmap_reshape_unsharded_dim(self):
        """vmap(reshape) where the reshaping dimension is NOT sharded."""
        batch = 4
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        def f(x):
            x_sharded = x.shard(mesh, P("tp", None))

            return reshape(x_sharded, (8, 4, 4))

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np_x.reshape(batch, 8, 4, 4)

        assert_shape(result, (batch, 8, 4, 4))
        assert_allclose(result, expected)

    def test_vmap_swap_axes_with_sharding(self):
        """vmap(swap_axes) with multi-axis sharding."""
        batch = 4
        mesh = DeviceMesh("mesh", (2, 2), ("dp", "tp"))

        def f(x):
            x_sharded = x.shard(mesh, P(None, "tp"))
            return swap_axes(x_sharded, 0, 1)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.swapaxes(np_x, 1, 2)

        assert_shape(result, (batch, 16, 8))
        assert_allclose(result, expected)


class TestVmapCompositeWithSharding:
    """Test vmap on composite functions with realistic sharding patterns."""

    def test_mlp_layer_megatron_style(self):
        """Full MLP layer: column parallel -> activation -> row parallel."""
        batch = 4
        hidden = 16
        ffn = 32
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        def mlp_layer(x, w1, w2, b1):

            w1_sharded = w1.shard(mesh, P(None, "tp"))
            h = matmul(x, w1_sharded)

            b1_sharded = b1.shard(mesh, P("tp"))
            h = add(h, b1_sharded)

            h = relu(h)

            w2_sharded = w2.shard(mesh, P("tp", None))
            out = matmul(h, w2_sharded)

            return out

        np_x = make_array(batch, hidden, seed=42)
        np_w1 = make_array(hidden, ffn, seed=43) * 0.1
        np_w2 = make_array(ffn, hidden, seed=44) * 0.1
        np_b1 = make_array(ffn, seed=45) * 0.1

        x = tensor_from_numpy(np_x)
        w1 = tensor_from_numpy(np_w1)
        w2 = tensor_from_numpy(np_w2)
        b1 = tensor_from_numpy(np_b1)

        vmapped_mlp = vmap(mlp_layer, in_axes=(0, None, None, None))

        print(f"\n{'='*70}")
        print("TEST: test_mlp_layer_megatron_style")
        print("Pattern: Column Parallel → ReLU → Row Parallel")
        print("  Layer 1: x @ w1_sharded(<*, tp>) = column parallel (no reduction)")
        print("  Layer 2: h @ w2_sharded(<tp, *>) = row parallel (AllReduce needed)")
        print("mesh_shape=(2,), mesh_axes=('tp',)")
        print(f"{'='*70}")
        t = trace(vmapped_mlp, x, w1, w2, b1)
        print(t)

        trace_str = str(t)
        if "all_reduce" in trace_str.lower():
            print("✅ AllReduce detected in trace - CORRECT for row-parallel!")
        else:
            print(
                "❌ WARNING: No AllReduce detected - row-parallel matmul may be INCORRECT!"
            )

        shard_count = trace_str.lower().count("shard(")
        print(f"📊 Found {shard_count} shard operations in trace")
        print(f"{'='*70}\n")

        result = vmapped_mlp(x, w1, w2, b1)

        h1 = np_x @ np_w1 + np_b1
        h1 = np.maximum(h1, 0)
        expected = h1 @ np_w2

        assert_shape(result, (batch, hidden))
        assert_allclose(result, expected, rtol=1e-4)

    def test_attention_qk_sharding(self):
        """Attention QK^T with sharding on hidden dimension."""
        batch = 2
        seq_len = 8
        hidden = 16
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        def compute_attention_scores(q, k):

            q_sharded = q.shard(mesh, P(None, "tp"))
            k_sharded = k.shard(mesh, P(None, "tp"))

            k_t = swap_axes(k_sharded, 0, 1)

            scores = matmul(q_sharded, k_t)
            return scores

        np_q = make_array(batch, seq_len, hidden, seed=42) * 0.1
        np_k = make_array(batch, seq_len, hidden, seed=43) * 0.1

        q = tensor_from_numpy(np_q)
        k = tensor_from_numpy(np_k)

        result = vmap(compute_attention_scores)(q, k)
        expected = np_q @ np_k.transpose(0, 2, 1)

        assert_shape(result, (batch, seq_len, seq_len))
        assert_allclose(result, expected, rtol=1e-4)


class TestNestedVmapWithSharding:
    """Test nested vmap with sharding."""

    def test_nested_vmap_with_sharding(self):
        """Doubly-nested vmap with sharding applied at inner level."""
        outer_batch = 2
        inner_batch = 4
        features = 16
        mesh = DeviceMesh("mesh", (2,), ("tp",))

        def inner_fn(x):
            x_sharded = x.shard(mesh, P("tp"))
            return relu(x_sharded)

        def outer_fn(x_batch):
            return vmap(inner_fn)(x_batch)

        np_x = make_array(outer_batch, inner_batch, features, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(outer_fn)(x)
        expected = np.maximum(np_x, 0)

        assert_shape(result, (outer_batch, inner_batch, features))
        assert_allclose(result, expected)


class TestVmapShardingAllUnaryOps:
    """Systematically test ALL unary ops with vmap+sharding."""

    UNARY_OPS = [
        ("relu", relu, lambda x: np.maximum(x, 0)),
        ("sigmoid", sigmoid, lambda x: 1 / (1 + np.exp(-x))),
        ("tanh", tanh, np.tanh),
        ("exp", exp, np.exp),
        ("neg", neg, lambda x: -x),
    ]

    @pytest.mark.parametrize("op_name,nabla_op,numpy_op", UNARY_OPS)
    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_unary_op_with_sharding(
        self, op_name, nabla_op, numpy_op, mesh_shape, mesh_axes
    ):
        """Test each unary op with sharding inside vmap."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return nabla_op(x_sharded)

        if op_name == "exp":
            np_x = make_array(batch, 8, 16, seed=42) * 0.1
        else:
            np_x = make_array(batch, 8, 16, seed=42)

        x = tensor_from_numpy(np_x)
        result = vmap(f)(x)
        expected = numpy_op(np_x)

        assert_shape(result, (batch, 8, 16))
        assert_allclose(result, expected, rtol=1e-4)


class TestVmapShardingAllBinaryOps:
    """Systematically test ALL binary ops with vmap+sharding."""

    BINARY_OPS = [
        ("add", add, lambda a, b: a + b),
        ("sub", sub, lambda a, b: a - b),
        ("mul", mul, lambda a, b: a * b),
        ("div", div, lambda a, b: a / b),
    ]

    @pytest.mark.parametrize("op_name,nabla_op,numpy_op", BINARY_OPS)
    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_binary_op_both_batched_sharded(
        self, op_name, nabla_op, numpy_op, mesh_shape, mesh_axes
    ):
        """Test binary op with both inputs batched and sharded."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x, y):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            y_sharded = y.shard(mesh, P(None, mesh_axes[-1]))
            return nabla_op(x_sharded, y_sharded)

        np_a = make_array(batch, 8, 16, seed=42)
        if op_name == "div":
            np_b = make_positive_array(batch, 8, 16, seed=43)
        else:
            np_b = make_array(batch, 8, 16, seed=43)

        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)

        result = vmap(f)(a, b)
        expected = numpy_op(np_a, np_b)

        assert_shape(result, (batch, 8, 16))
        assert_allclose(result, expected)

    @pytest.mark.parametrize("op_name,nabla_op,numpy_op", BINARY_OPS)
    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_binary_op_batched_broadcast(
        self, op_name, nabla_op, numpy_op, mesh_shape, mesh_axes
    ):
        """Test binary op with batched input + broadcast weight (realistic pattern)."""
        batch = 4
        features = 16
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x, w):
            w_sharded = w.shard(mesh, P(mesh_axes[-1]))
            return nabla_op(x, w_sharded)

        np_x = make_array(batch, features, seed=42)
        if op_name == "div":
            np_w = make_positive_array(features, seed=43)
        else:
            np_w = make_array(features, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        result = vmap(f, in_axes=(0, None))(x, w)
        expected = numpy_op(np_x, np_w)

        assert_shape(result, (batch, features))
        assert_allclose(result, expected)


class TestVmapShardingAllViewOps:
    """Systematically test ALL view ops with vmap+sharding."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_reshape_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(reshape) with sharded input - merge pattern.

        Tests reshape that merges a sharded dimension. The library should
        auto-gather to maintain correctness.
        """
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return reshape(x_sharded, (128,))

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np_x.reshape(batch, 128)

        assert_shape(result, (batch, 128))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_squeeze_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(squeeze) with sharded input."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return squeeze(x_sharded, axis=0)

        np_x = make_array(batch, 1, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np_x.squeeze(axis=1)

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_unsqueeze_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(unsqueeze) with sharded input."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(mesh_axes[-1]))
            return unsqueeze(x_sharded, axis=0)

        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np_x[:, np.newaxis, :]

        assert_shape(result, (batch, 1, 16))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_swap_axes_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(swap_axes) with sharded input."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return swap_axes(x_sharded, 0, 1)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.swapaxes(np_x, 1, 2)

        assert_shape(result, (batch, 16, 8))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_broadcast_to_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(broadcast_to) with sharded input."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return broadcast_to(x_sharded, (8, 16))

        np_x = make_array(batch, 1, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.broadcast_to(np_x, (batch, 8, 16))

        assert_shape(result, (batch, 8, 16))
        assert_allclose(result, expected)


class TestVmapShardingAllReductionOps:
    """Systematically test ALL reduction ops with vmap+sharding."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_reduce_sum_axis0(self, mesh_shape, mesh_axes):
        """vmap(reduce_sum) reducing first logical axis."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return reduce_sum(x_sharded, axis=0)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.sum(np_x, axis=1)

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_reduce_sum_axis1(self, mesh_shape, mesh_axes):
        """vmap(reduce_sum) reducing sharded axis (requires AllReduce)."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return reduce_sum(x_sharded, axis=1)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.sum(np_x, axis=2)

        assert_shape(result, (batch, 8))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_mean_axis0(self, mesh_shape, mesh_axes):
        """vmap(mean) reducing first logical axis."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return mean(x_sharded, axis=0)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.mean(np_x, axis=1)

        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_mean_axis1(self, mesh_shape, mesh_axes):
        """vmap(mean) reducing sharded axis."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return mean(x_sharded, axis=1)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected = np.mean(np_x, axis=2)

        assert_shape(result, (batch, 8))
        assert_allclose(result, expected)


class TestVmapShardingAllMatmulPatterns:
    """Systematically test matmul with all sharding patterns."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_matmul_column_parallel(self, mesh_shape, mesh_axes):
        """Matmul with weight sharded on output (column parallel)."""
        batch = 4
        M, K, N = 8, 16, 32
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x, w):
            w_sharded = w.shard(mesh, P(None, mesh_axes[-1]))
            return matmul(x, w_sharded)

        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        result = vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w

        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((4,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_matmul_row_parallel(self, mesh_shape, mesh_axes):
        """Matmul with weight sharded on input (row parallel) - requires AllReduce."""
        batch = 4
        M, K, N = 8, 32, 16
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x, w):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            w_sharded = w.shard(mesh, P(mesh_axes[-1], None))
            return matmul(x_sharded, w_sharded)

        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)

        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)

        result = vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w

        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_matmul_both_weights_sharded(self, mesh_shape, mesh_axes):
        """Two consecutive matmuls with Megatron-style sharding."""
        batch = 4
        H, FFN = 16, 32
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x, w1, w2):
            w1_sharded = w1.shard(mesh, P(None, mesh_axes[-1]))
            h = matmul(x, w1_sharded)

            w2_sharded = w2.shard(mesh, P(mesh_axes[-1], None))
            out = matmul(h, w2_sharded)
            return out

        np_x = make_array(batch, H, seed=42)
        np_w1 = make_array(H, FFN, seed=43)
        np_w2 = make_array(FFN, H, seed=44)

        x = tensor_from_numpy(np_x)
        w1 = tensor_from_numpy(np_w1)
        w2 = tensor_from_numpy(np_w2)

        result = vmap(f, in_axes=(0, None, None))(x, w1, w2)
        expected = (np_x @ np_w1) @ np_w2

        assert_shape(result, (batch, H))
        assert_allclose(result, expected, rtol=1e-4)


class TestVmapShardingMultiOutputOps:
    """Test multi-output ops with vmap+sharding."""

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_split_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(split) with sharded input."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(mesh_axes[-1]))
            return split(x_sharded, num_splits=2, axis=0)

        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected_parts = np.split(np_x, 2, axis=1)

        assert len(result) == 2
        for i, (r, e) in enumerate(zip(result, expected_parts, strict=False)):
            assert_shape(r, (batch, 8))
            assert_allclose(r, e)

    @pytest.mark.parametrize(
        "mesh_shape,mesh_axes",
        [
            ((2,), ("tp",)),
            ((2, 2), ("dp", "tp")),
        ],
    )
    def test_chunk_with_sharding(self, mesh_shape, mesh_axes):
        """vmap(chunk) with sharded input."""
        batch = 4
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(None, mesh_axes[-1]))
            return chunk(x_sharded, chunks=4, axis=0)

        np_x = make_array(batch, 8, 16, seed=42)
        x = tensor_from_numpy(np_x)

        result = vmap(f)(x)
        expected_chunks = np.array_split(np_x, 4, axis=1)

        assert len(result) == 4
        for i, (r, e) in enumerate(zip(result, expected_chunks, strict=False)):
            assert_shape(r, (batch, 2, 16))
            assert_allclose(r, e)
