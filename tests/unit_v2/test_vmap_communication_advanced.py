# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Advanced stress tests for vmap + communication operations.

Covers:
- Nested vmap (batch_dims > 1)
- vmap with in_axes != 0
- Gradients through vmapped communication
- Complex mesh patterns
"""

import jax
import jax.numpy as jnp
import pytest

import nabla as nb
from nabla.core.sharding.spec import DeviceMesh, P
from nabla.ops.communication import all_gather, reduce_scatter, all_reduce
from .common import (
    assert_allclose,
    tensor_from_jax,
)


class TestVmapCommunicationAdvanced:

    def test_nested_vmap_all_gather(self):
        """vmap(vmap(all_gather)): batch_dims=2."""
        # 2 batch dimensions: B1, B2.
        # Inside, we shard a hidden dimension and gather it.
        # This checks if axis shifting works with batch_dims=2.

        B1, B2, H = 2, 2, 4
        mesh = DeviceMesh("mesh_nested", (2,), ("tp",))

        def inner_f(x):
            # x has shape (H,).
            # We shard it on 'tp' and gather back.
            x_sharded = x.shard(mesh, P("tp"))
            # axis=0 of x refers to H.
            # Physically, x has 2 batch dims: (B1, B2, H).
            # So axis should shift by 2.
            return all_gather(x_sharded, axis=0)

        def outer_f(x):
            return nb.vmap(inner_f)(x)

        np_x = jax.random.normal(jax.random.PRNGKey(0), (B1, B2, H), dtype=jnp.float32)
        x = tensor_from_jax(np_x)

        # Apply nested vmap
        # vmap over B1, then vmap over B2
        result = nb.vmap(outer_f)(x)

        # Result should be replicated and match input
        assert_allclose(result, np_x)
        assert tuple(int(d) for d in result.shape) == (B1, B2, H)

    def test_vmap_in_axes_shift(self):
        """vmap(all_gather, in_axes=1)."""
        # Data shape: (H, B). Batch dim is at index 1.
        # vmap will move it to 0 (physically).
        # We perform all_gather on H (axis 0).
        # Check if shifting logic holds when input axis was permuted.

        H, B = 4, 2
        mesh = DeviceMesh("mesh_in_axes", (2,), ("tp",))

        def f(x):
            # x shape (H,).
            # Shard H on 'tp'.
            x_sharded = x.shard(mesh, P("tp"))
            return all_gather(x_sharded, axis=0)

        np_x = jax.random.normal(jax.random.PRNGKey(1), (H, B), dtype=jnp.float32)
        x = tensor_from_jax(np_x)

        # vmap over axis 1
        result = nb.vmap(f, in_axes=1, out_axes=1)(x)

        assert_allclose(result, np_x)
        assert tuple(int(d) for d in result.shape) == (H, B)

    def test_grad_vmap_all_reduce(self):
        """grad(vmap(all_reduce))."""
        # Check if VJP works through vmapped communication.
        # all_reduce VJP is identity (for sum).

        B, H = 2, 4
        mesh = DeviceMesh("mesh_grad", (2,), ("tp",))

        def loss_fn(x):
            def body(u):
                # u: (H,)
                u_sharded = u.shard(mesh, P("tp"))
                # all_reduce(sum) -> replicated sum
                res = all_reduce(u_sharded, reduce_op="sum")
                return nb.reduce_sum(res * res)

            # vmap over batch
            batch_losses = nb.vmap(body)(x)
            return nb.reduce_sum(batch_losses)

        np_x = jax.random.normal(jax.random.PRNGKey(2), (B, H), dtype=jnp.float32)
        x = tensor_from_jax(np_x)

        # Nabla Grad
        grad_fn = nb.grad(loss_fn)
        grad_x = grad_fn(x)

        mid = H // 2
        x_L = np_x[:, :mid]
        x_R = np_x[:, mid:]
        res = x_L + x_R
        # Gradient is concatenation of gradients for each part
        grad_L = 2 * res
        grad_R = 2 * res
        expected_grad = jnp.concatenate([grad_L, grad_R], axis=1)

        assert_allclose(grad_x, expected_grad)

    def test_vmap_reduce_scatter_grad(self):
        """grad(vmap(reduce_scatter))."""
        # reduce_scatter VJP involves all_gather (broadcasting gradients).

        B, H = 2, 4
        mesh = DeviceMesh("mesh_grad_rs", (2,), ("tp",))

        def loss_fn(x):
            def body(u):
                # u: (H,) Replicated
                u_rep = u.shard(mesh, P(None))
                # reduce_scatter(axis=0) -> (H/2,) Sharded
                res = reduce_scatter(u_rep, axis=0)
                # Sum squares of shards
                return nb.reduce_sum(res * res)

            return nb.reduce_sum(nb.vmap(body)(x))

        np_x = jax.random.normal(jax.random.PRNGKey(3), (B, H), dtype=jnp.float32)
        x = tensor_from_jax(np_x)

        grad_x = nb.grad(loss_fn)(x)

        # JAX Reference
        def jax_loss(x_arr):
            return 0.0

        # We verify shapes and finiteness for now
        assert tuple(int(d) for d in grad_x.shape) == (B, H)
        from .common import to_jax

        assert not jnp.isnan(to_jax(grad_x)).any()
