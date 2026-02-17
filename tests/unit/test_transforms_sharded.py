# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Sharded VJP/JVP tests - verify transforms work with SPMD sharding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from tests.unit.common import (
    MESH_CONFIGS,
    DeviceMesh,
    cleanup_caches,
    make_jax_array,
    replicated,
    shard_on_axis,
    tensor_from_jax,
    to_jax,
)


def _close(nb_val, jax_val, rtol=1e-4, atol=1e-4):
    """Assert nabla Tensor ≈ JAX array."""
    np.testing.assert_allclose(to_jax(nb_val), jax_val, rtol=rtol, atol=atol)


# ═════════════════════════════════════════════════════════════════════════════
#  SHARDED VJP
# ═════════════════════════════════════════════════════════════════════════════


class TestShardedVJP:
    """VJP with sharded tensors."""

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:3])
    def test_vjp_sharded_input(self, mesh_name, mesh_shape, axis_names):
        """VJP with input sharded along first axis."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(8, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)
        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)

        def f(x):
            return nb.reduce_sum(nb.mul(x, x))

        # VJP with sharded input
        out_nb, vjp_fn = nb.vjp(f, x_sharded)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        # JAX reference
        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(x * x), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:1])
    def test_vjp_replicated_input(self, mesh_name, mesh_shape, axis_names):
        """VJP with replicated input."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(4, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)
        x_replicated = replicated(x_nb, mesh)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.exp(x)), x_replicated)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(jnp.exp(x)), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:2])
    def test_vjp_binary_sharded(self, mesh_name, mesh_shape, axis_names):
        """VJP with binary op on sharded tensors."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(8, 4, seed=1)
        y_jax = make_jax_array(8, 4, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)
        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)
        y_sharded = shard_on_axis(y_nb, mesh, axis=0, mesh_axis=0)

        out_nb, vjp_fn = nb.vjp(
            lambda x, y: nb.reduce_sum(nb.mul(x, y)), x_sharded, y_sharded
        )
        gx_nb, gy_nb = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x, y: jnp.sum(x * y), x_jax, y_jax)
        gx_jax, gy_jax = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gy_nb, gy_jax)

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:1])
    def test_vjp_sharded_matmul(self, mesh_name, mesh_shape, axis_names):
        """VJP of matmul with sharded inputs."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(8, 4, seed=1)
        w_jax = make_jax_array(4, 6, seed=2)
        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)
        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)
        w_replicated = replicated(w_nb, mesh)

        out_nb, vjp_fn = nb.vjp(nb.matmul, x_sharded, w_replicated)
        gx_nb, gw_nb = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(jnp.matmul, x_jax, w_jax)
        gx_jax, gw_jax = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gw_nb, gw_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  SHARDED JVP
# ═════════════════════════════════════════════════════════════════════════════


class TestShardedJVP:
    """JVP with sharded tensors."""

    @pytest.mark.xfail(reason="all_reduce does not implement jvp_rule")
    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:3])
    def test_jvp_sharded_input(self, mesh_name, mesh_shape, axis_names):
        """JVP with input sharded along first axis."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(8, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)
        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)
        t_sharded = shard_on_axis(t_nb, mesh, axis=0, mesh_axis=0)

        def f(x):
            return nb.reduce_sum(nb.mul(x, x))

        out_nb, tan_nb = nb.jvp(f, (x_sharded,), (t_sharded,))
        out_jax, tan_jax = jax.jvp(lambda x: jnp.sum(x * x), (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:1])
    def test_jvp_replicated_input(self, mesh_name, mesh_shape, axis_names):
        """JVP with replicated input."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(4, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)
        x_replicated = replicated(x_nb, mesh)
        t_replicated = replicated(t_nb, mesh)

        out_nb, tan_nb = nb.jvp(
            lambda x: nb.reduce_sum(nb.exp(x)), (x_replicated,), (t_replicated,)
        )
        out_jax, tan_jax = jax.jvp(lambda x: jnp.sum(jnp.exp(x)), (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    @pytest.mark.xfail(reason="all_reduce does not implement jvp_rule")
    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:2])
    def test_jvp_binary_sharded(self, mesh_name, mesh_shape, axis_names):
        """JVP with binary op on sharded tensors."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(8, 4, seed=1)
        y_jax = make_jax_array(8, 4, seed=2)
        tx_jax = jnp.ones_like(x_jax)
        ty_jax = jnp.ones_like(y_jax)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)
        tx_nb = tensor_from_jax(tx_jax)
        ty_nb = tensor_from_jax(ty_jax)

        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)
        y_sharded = shard_on_axis(y_nb, mesh, axis=0, mesh_axis=0)
        tx_sharded = shard_on_axis(tx_nb, mesh, axis=0, mesh_axis=0)
        ty_sharded = shard_on_axis(ty_nb, mesh, axis=0, mesh_axis=0)

        out_nb, tan_nb = nb.jvp(
            lambda x, y: nb.reduce_sum(nb.mul(x, y)),
            (x_sharded, y_sharded),
            (tx_sharded, ty_sharded),
        )
        out_jax, tan_jax = jax.jvp(
            lambda x, y: jnp.sum(x * y), (x_jax, y_jax), (tx_jax, ty_jax)
        )

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  VMAP + SHARDING
# ═════════════════════════════════════════════════════════════════════════════


class TestVmapShardedTransforms:
    """vmap composed with sharded vjp/jvp."""

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:1])
    def test_vmap_vjp_sharded(self, mesh_name, mesh_shape, axis_names):
        """vmap(vjp(...)) with sharded tensors."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(4, 3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)
        # Shard on batch dim
        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)

        def vjp_grad_nb(x):
            out, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.mul(x, x)), x)
            (g,) = vjp_fn(nb.ones_like(out))
            return g

        def vjp_grad_jax(x):
            out, vjp_fn = jax.vjp(lambda x: jnp.sum(x * x), x)
            (g,) = vjp_fn(jnp.ones_like(out))
            return g

        nb_res = nb.vmap(vjp_grad_nb)(x_sharded)
        jax_res = jax.vmap(vjp_grad_jax)(x_jax)

        _close(nb_res, jax_res)

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:1])
    def test_vmap_jvp_sharded(self, mesh_name, mesh_shape, axis_names):
        """vmap(jvp(...)) with sharded tensors."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(4, 3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)
        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)
        t_sharded = shard_on_axis(t_nb, mesh, axis=0, mesh_axis=0)

        def jvp_fn_nb(x, t):
            out, tan = nb.jvp(lambda x: nb.reduce_sum(nb.mul(x, x)), (x,), (t,))
            return tan

        def jvp_fn_jax(x, t):
            out, tan = jax.jvp(lambda x: jnp.sum(x * x), (x,), (t,))
            return tan

        nb_res = nb.vmap(jvp_fn_nb)(x_sharded, t_sharded)
        jax_res = jax.vmap(jvp_fn_jax)(x_jax, t_jax)

        _close(nb_res, jax_res)


# ═════════════════════════════════════════════════════════════════════════════
#  REDUCTION OPS WITH SHARDING
# ═════════════════════════════════════════════════════════════════════════════


class TestShardedReductions:
    """VJP/JVP with reductions and sharding."""

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:2])
    def test_vjp_reduce_sum_sharded(self, mesh_name, mesh_shape, axis_names):
        """VJP of reduce_sum with sharded input."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(8, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)
        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)

        out_nb, vjp_fn = nb.vjp(nb.reduce_sum, x_sharded)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(jnp.sum, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    @pytest.mark.parametrize("mesh_name,mesh_shape,axis_names", MESH_CONFIGS[:1])
    def test_vjp_reduce_sum_axis_sharded(self, mesh_name, mesh_shape, axis_names):
        """VJP of reduce_sum along axis with sharded input."""
        cleanup_caches()
        mesh = DeviceMesh(mesh_name, mesh_shape, axis_names)

        x_jax = make_jax_array(8, 4, 6, seed=1)
        x_nb = tensor_from_jax(x_jax)
        x_sharded = shard_on_axis(x_nb, mesh, axis=0, mesh_axis=0)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(x, axis=1), x_sharded)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(x, axis=1), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)
