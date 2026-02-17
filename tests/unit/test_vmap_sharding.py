# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Test vmap with sharding - realistic patterns from production ML workloads."""

import jax
import jax.numpy as jnp
import pytest

import nabla as nb
from nabla.core.sharding.spec import DeviceMesh, P

from .common import (
    MESH_CONFIGS,
    assert_allclose,
    tensor_from_jax,
)


class TestVmapShardingUnary:
    """Test vmap(unary_op) with sharding inside function."""

    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", MESH_CONFIGS[:3])
    def test_relu_sharded_features(self, mesh_name, mesh_shape, mesh_axes):
        """vmap(relu) with feature dimension sharded (tensor parallelism)."""
        batch, features = 4, 16
        mesh = DeviceMesh(f"mesh_relu_{mesh_name}", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(mesh_axes[-1]))
            return nb.relu(x_sharded)

        np_x = jax.random.normal(
            jax.random.PRNGKey(42), (batch, features), dtype=jnp.float32
        )
        x = tensor_from_jax(np_x)

        result = nb.vmap(f)(x)
        expected = jnp.maximum(np_x, 0)

        assert tuple(int(d) for d in result.shape) == (batch, features)
        assert_allclose(result, expected)


class TestVmapShardingBinary:
    """Test vmap(binary_op) with realistic batched+broadcast patterns."""

    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", MESH_CONFIGS[:3])
    def test_add_batched_data_sharded_bias(self, mesh_name, mesh_shape, mesh_axes):
        """vmap(add) with batched input + broadcast sharded bias."""
        batch, hidden = 4, 16
        mesh = DeviceMesh(f"mesh_add_{mesh_name}", mesh_shape, mesh_axes)

        def f(x, bias):
            bias_sharded = bias.shard(mesh, P(mesh_axes[-1]))
            return nb.add(x, bias_sharded)

        np_x = jax.random.normal(
            jax.random.PRNGKey(43), (batch, hidden), dtype=jnp.float32
        )
        np_bias = jax.random.normal(
            jax.random.PRNGKey(44), (hidden,), dtype=jnp.float32
        )

        x = tensor_from_jax(np_x)
        bias = tensor_from_jax(np_bias)

        result = nb.vmap(f, in_axes=(0, None))(x, bias)
        expected = np_x + np_bias

        assert tuple(int(d) for d in result.shape) == (batch, hidden)
        assert_allclose(result, expected)


class TestVmapShardingMatmul:
    """Test vmap(matmul) with Megatron-style column/row parallel patterns."""

    def test_column_parallel(self):
        """Column-parallel matmul: shard output dimension."""
        batch, in_feat, out_feat = 4, 16, 32
        mesh = DeviceMesh("mesh_matmul_col", (2,), ("tp",))

        def f(x, w):
            w_sharded = w.shard(mesh, P(None, "tp"))
            return nb.matmul(x, w_sharded)

        np_x = jax.random.normal(
            jax.random.PRNGKey(45), (batch, in_feat), dtype=jnp.float32
        )
        np_w = jax.random.normal(
            jax.random.PRNGKey(46), (in_feat, out_feat), dtype=jnp.float32
        )

        x = tensor_from_jax(np_x)
        w = tensor_from_jax(np_w)

        result = nb.vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w

        assert tuple(int(d) for d in result.shape) == (batch, out_feat)
        assert_allclose(result, expected, rtol=1e-4)

    def test_row_parallel(self):
        """Row-parallel matmul: shard contracting dimension (needs AllReduce)."""
        batch, in_feat, out_feat = 4, 32, 16
        mesh = DeviceMesh("mesh_matmul_row", (2,), ("tp",))

        def f(x, w):
            x_sharded = x.shard(mesh, P("tp"))
            w_sharded = w.shard(mesh, P("tp", None))
            return nb.matmul(x_sharded, w_sharded)

        np_x = jax.random.normal(
            jax.random.PRNGKey(47), (batch, in_feat), dtype=jnp.float32
        )
        np_w = jax.random.normal(
            jax.random.PRNGKey(48), (in_feat, out_feat), dtype=jnp.float32
        )

        x = tensor_from_jax(np_x)
        w = tensor_from_jax(np_w)

        result = nb.vmap(f, in_axes=(0, None))(x, w)
        expected = np_x @ np_w

        assert tuple(int(d) for d in result.shape) == (batch, out_feat)
        assert_allclose(result, expected, rtol=1e-4)


class TestVmapShardingReduction:
    """Test vmap(reduction) with sharded inputs."""

    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", MESH_CONFIGS[:2])
    def test_reduce_sum_sharded_axis(self, mesh_name, mesh_shape, mesh_axes):
        """vmap(reduce_sum) reducing over sharded dimension (needs AllReduce)."""

        batch = 4
        mesh = DeviceMesh(f"mesh_reduce_{mesh_name}", mesh_shape, mesh_axes)

        def f(x):
            x_sharded = x.shard(mesh, P(mesh_axes[-1]))
            return nb.reduce_sum(x_sharded, axis=0)

        np_x = jax.random.normal(jax.random.PRNGKey(49), (batch, 16), dtype=jnp.float32)
        x = tensor_from_jax(np_x)

        result = nb.vmap(f)(x)
        expected = jnp.sum(np_x, axis=1)

        assert tuple(int(d) for d in result.shape) == (batch,)
        assert_allclose(result, expected, rtol=1e-5)


class TestVmapShardingComposite:
    """Test realistic composite functions with vmap+sharding."""

    def test_mlp_layer_megatron(self):
        """Full MLP: column parallel → ReLU → row parallel."""
        batch, hidden, ffn = 4, 16, 32
        mesh = DeviceMesh("mesh_mlp", (2,), ("tp",))

        def mlp(x, w1, w2, b1):
            w1_sharded = w1.shard(mesh, P(None, "tp"))
            h = nb.matmul(x, w1_sharded)

            b1_sharded = b1.shard(mesh, P("tp"))
            h = nb.add(h, b1_sharded)

            h = nb.relu(h)

            w2_sharded = w2.shard(mesh, P("tp", None))
            out = nb.matmul(h, w2_sharded)

            return out

        np_x = (
            jax.random.normal(
                jax.random.PRNGKey(50), (batch, hidden), dtype=jnp.float32
            )
            * 0.1
        )
        np_w1 = (
            jax.random.normal(jax.random.PRNGKey(51), (hidden, ffn), dtype=jnp.float32)
            * 0.1
        )
        np_w2 = (
            jax.random.normal(jax.random.PRNGKey(52), (ffn, hidden), dtype=jnp.float32)
            * 0.1
        )
        np_b1 = (
            jax.random.normal(jax.random.PRNGKey(53), (ffn,), dtype=jnp.float32) * 0.1
        )

        x = tensor_from_jax(np_x)
        w1 = tensor_from_jax(np_w1)
        w2 = tensor_from_jax(np_w2)
        b1 = tensor_from_jax(np_b1)

        result = nb.vmap(mlp, in_axes=(0, None, None, None))(x, w1, w2, b1)

        h1 = np_x @ np_w1 + np_b1
        h1 = jnp.maximum(h1, 0)
        expected = h1 @ np_w2

        assert tuple(int(d) for d in result.shape) == (batch, hidden)
        assert_allclose(result, expected, rtol=1e-4)
