# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Rigorous stress tests for communication operations under complex conditions."""

import jax
import jax.numpy as jnp
import pytest
import nabla as nb
from nabla.core.sharding.spec import DeviceMesh, P
from nabla.ops.communication import all_reduce, reduce_scatter, all_gather, all_to_all
from .common import assert_allclose, tensor_from_jax, to_jax

class TestCommunicationRigorous:

    def test_nested_vmap_all_reduce_grad(self):
        """Gradient of nested vmap(vmap(all_reduce))."""
        B1, B2, H = 2, 2, 4
        mesh = DeviceMesh("mesh_nested_ar", (2,), ("tp",))
        
        def loss_fn(x):
            def inner(u):
                u_sharded = u.shard(mesh, P("tp"))
                res = all_reduce(u_sharded, reduce_op="sum")
                return nb.reduce_sum(res * res)
            
            return nb.reduce_sum(nb.vmap(nb.vmap(inner))(x))
            
        np_x = jax.random.normal(jax.random.PRNGKey(42), (B1, B2, H), dtype=jnp.float32)
        x = tensor_from_jax(np_x)
        
        grad_x = nb.grad(loss_fn)(x)
        
        np_res = np_x[:, :, :2] + np_x[:, :, 2:]
        expected_grad = jnp.concatenate([2 * np_res, 2 * np_res], axis=2)
        
        assert_allclose(grad_x, expected_grad)

    def test_vmap_reduce_scatter_multidim(self):
        """vmap(reduce_scatter) on a 2D mesh."""        
        B, H = 2, 8
        mesh = DeviceMesh("mesh_2d_rs", (2, 2), ("x", "y"))
        
        def f(u):
            u_s = u.shard(mesh, P("x"))
            return reduce_scatter(u_s, axis=0)
            
        np_x = jax.random.normal(jax.random.PRNGKey(43), (B, H), dtype=jnp.float32)
        x = tensor_from_jax(np_x)
        
        res = nb.vmap(f)(x)
    
        assert tuple(int(d) for d in res.shape) == (B, 4)
        
        spec = res.sharding
        assert spec.dim_specs[0].is_replicated()
        axes = spec.dim_specs[1].axes
        assert set(axes) == {"x", "y"}

    def test_all_reduce_sum_grad_simple(self):
        """Simpler all_reduce grad test without vmap."""
        H = 4
        mesh = DeviceMesh("mesh_simple_ar", (2,), ("tp",))
        
        def loss_fn(u):
            u_s = u.shard(mesh, P("tp"))
            res = all_reduce(u_s) 
            return nb.reduce_sum(res * res)
            
        np_u = jax.random.normal(jax.random.PRNGKey(44), (H,), dtype=jnp.float32)
        u = tensor_from_jax(np_u)
        
        grad_u = nb.grad(loss_fn)(u)
        
        res_val = np_u[:2] + np_u[2:]
        expected = jnp.concatenate([2 * res_val, 2 * res_val])
        
        assert_allclose(grad_u, expected)

