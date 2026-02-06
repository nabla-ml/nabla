# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Transform composition tests: vmap+vjp/jvp, nested transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from tests.unit.common import (
    cleanup_caches,
    make_jax_array,
    make_positive_jax_array,
    tensor_from_jax,
    to_jax,
)


def _close(nb_val, jax_val, rtol=1e-4, atol=1e-4):
    """Assert nabla Tensor ≈ JAX array."""
    np.testing.assert_allclose(to_jax(nb_val), jax_val, rtol=rtol, atol=atol)


# ═════════════════════════════════════════════════════════════════════════════
#  VMAP + VJP
# ═════════════════════════════════════════════════════════════════════════════

class TestVmapVJP:
    """Composition of vmap with vjp."""

    def test_vmap_vjp_basic(self):
        """vmap(vjp(f)) over batched inputs."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        def vjp_grad_nb(x):
            out, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.mul(x, x)), x)
            (g,) = vjp_fn(nb.ones_like(out))
            return g

        def vjp_grad_jax(x):
            out, vjp_fn = jax.vjp(lambda x: jnp.sum(x * x), x)
            (g,) = vjp_fn(jnp.ones_like(out))
            return g

        nb_res = nb.vmap(vjp_grad_nb)(x_nb)
        jax_res = jax.vmap(vjp_grad_jax)(x_jax)

        _close(nb_res, jax_res)

    def test_vmap_vjp_binary(self):
        """vmap(vjp(f)) where f takes two arguments."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        y_jax = make_jax_array(3, 4, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        def vjp_grad_nb(x, y):
            out, vjp_fn = nb.vjp(lambda x, y: nb.reduce_sum(nb.mul(x, y)), x, y)
            gx, gy = vjp_fn(nb.ones_like(out))
            return gx

        def vjp_grad_jax(x, y):
            out, vjp_fn = jax.vjp(lambda x, y: jnp.sum(x * y), x, y)
            gx, gy = vjp_fn(jnp.ones_like(out))
            return gx

        nb_res = nb.vmap(vjp_grad_nb)(x_nb, y_nb)
        jax_res = jax.vmap(vjp_grad_jax)(x_jax, y_jax)

        _close(nb_res, jax_res)

    def test_vmap_vjp_unary_ops(self):
        """vmap(vjp(unary_op))."""
        cleanup_caches()
        x_jax = make_positive_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        def vjp_grad_nb(x):
            out, vjp_fn = nb.vjp(nb.exp, x)
            (g,) = vjp_fn(nb.ones_like(out))
            return g

        def vjp_grad_jax(x):
            out, vjp_fn = jax.vjp(jnp.exp, x)
            (g,) = vjp_fn(jnp.ones_like(out))
            return g

        nb_res = nb.vmap(vjp_grad_nb)(x_nb)
        jax_res = jax.vmap(vjp_grad_jax)(x_jax)

        _close(nb_res, jax_res)

    def test_vmap_vjp_matmul(self):
        """vmap over batch dim, vjp computes per-batch gradients."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, 4, seed=1)
        w_jax = make_jax_array(4, 5, seed=2)
        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)

        def vjp_grad_nb(x):
            out, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.matmul(x, w_nb)), x)
            (g,) = vjp_fn(nb.ones_like(out))
            return g

        def vjp_grad_jax(x):
            out, vjp_fn = jax.vjp(lambda x: jnp.sum(x @ w_jax), x)
            (g,) = vjp_fn(jnp.ones_like(out))
            return g

        nb_res = nb.vmap(vjp_grad_nb)(x_nb)
        jax_res = jax.vmap(vjp_grad_jax)(x_jax)

        _close(nb_res, jax_res)


# ═════════════════════════════════════════════════════════════════════════════
#  VMAP + JVP
# ═════════════════════════════════════════════════════════════════════════════

class TestVmapJVP:
    """Composition of vmap with jvp."""

    def test_vmap_jvp_basic(self):
        """vmap(jvp(f)) over batched inputs."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        def jvp_fn_nb(x, t):
            out, tan = nb.jvp(lambda x: nb.reduce_sum(nb.mul(x, x)), (x,), (t,))
            return tan

        def jvp_fn_jax(x, t):
            out, tan = jax.jvp(lambda x: jnp.sum(x * x), (x,), (t,))
            return tan

        nb_res = nb.vmap(jvp_fn_nb)(x_nb, t_nb)
        jax_res = jax.vmap(jvp_fn_jax)(x_jax, t_jax)

        _close(nb_res, jax_res)

    def test_vmap_jvp_binary(self):
        """vmap(jvp(f)) where f takes two arguments."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        y_jax = make_jax_array(3, 4, seed=2)
        tx_jax = jnp.ones_like(x_jax)
        ty_jax = jnp.ones_like(y_jax)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)
        tx_nb = tensor_from_jax(tx_jax)
        ty_nb = tensor_from_jax(ty_jax)

        def jvp_fn_nb(x, y, tx, ty):
            out, tan = nb.jvp(lambda x, y: nb.reduce_sum(nb.mul(x, y)), (x, y), (tx, ty))
            return tan

        def jvp_fn_jax(x, y, tx, ty):
            out, tan = jax.jvp(lambda x, y: jnp.sum(x * y), (x, y), (tx, ty))
            return tan

        nb_res = nb.vmap(jvp_fn_nb)(x_nb, y_nb, tx_nb, ty_nb)
        jax_res = jax.vmap(jvp_fn_jax)(x_jax, y_jax, tx_jax, ty_jax)

        _close(nb_res, jax_res)

    def test_vmap_jvp_unary_ops(self):
        """vmap(jvp(unary_op))."""
        cleanup_caches()
        x_jax = make_positive_jax_array(3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        def jvp_fn_nb(x, t):
            out, tan = nb.jvp(nb.exp, (x,), (t,))
            return tan

        def jvp_fn_jax(x, t):
            out, tan = jax.jvp(jnp.exp, (x,), (t,))
            return tan

        nb_res = nb.vmap(jvp_fn_nb)(x_nb, t_nb)
        jax_res = jax.vmap(jvp_fn_jax)(x_jax, t_jax)

        _close(nb_res, jax_res)


# ═════════════════════════════════════════════════════════════════════════════
#  NESTED VMAP
# ═════════════════════════════════════════════════════════════════════════════

class TestNestedVmap:
    """Nested vmap compositions."""

    def test_vmap_vmap_basic(self):
        """vmap(vmap(f))."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        nb_res = nb.vmap(nb.vmap(nb.exp))(x_nb)
        jax_res = jax.vmap(jax.vmap(jnp.exp))(x_jax)

        _close(nb_res, jax_res)

    def test_vmap_vmap_binary(self):
        """vmap(vmap(binary_op))."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, 4, seed=1)
        y_jax = make_jax_array(2, 3, 4, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        nb_res = nb.vmap(nb.vmap(nb.add))(x_nb, y_nb)
        jax_res = jax.vmap(jax.vmap(jnp.add))(x_jax, y_jax)

        _close(nb_res, jax_res)


# ═════════════════════════════════════════════════════════════════════════════
#  VJP + JVP consistency
# ═════════════════════════════════════════════════════════════════════════════

class TestVJPJVPConsistency:
    """Verify VJP and JVP give consistent Jacobian information."""

    def test_vjp_jvp_scalar(self):
        """For scalar f, vjp and jvp should match."""
        cleanup_caches()
        x_jax = jnp.array(2.0)
        x_nb = tensor_from_jax(x_jax)

        def f(x):
            return nb.exp(nb.sin(x))

        # JVP
        _, tan = nb.jvp(f, (x_nb,), (tensor_from_jax(jnp.array(1.0)),))

        # VJP
        _, vjp_fn = nb.vjp(f, x_nb)
        (grad,) = vjp_fn(tensor_from_jax(jnp.array(1.0)))

        _close(tan, to_jax(grad))

    def test_vjp_jvp_vector(self):
        """For R^n -> R, gradients should match directional derivatives."""
        cleanup_caches()
        x_jax = make_jax_array(4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        def f(x):
            return nb.reduce_sum(nb.mul(x, x))

        # VJP gives full gradient
        _, vjp_fn = nb.vjp(f, x_nb)
        (grad,) = vjp_fn(tensor_from_jax(jnp.array(1.0)))

        # JVP with identity directions should sum to gradient
        directions = jnp.eye(4)
        jvp_results = []
        for i in range(4):
            _, tan = nb.jvp(f, (x_nb,), (tensor_from_jax(directions[i]),))
            jvp_results.append(to_jax(tan))

        # Each JVP gives partial derivative
        # grad[i] should equal jvp_results[i]
        for i in range(4):
            np.testing.assert_allclose(
                to_jax(grad)[i], jvp_results[i], rtol=1e-4, atol=1e-4
            )


# ═════════════════════════════════════════════════════════════════════════════
#  HIGHER-ORDER COMPOSITIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestHigherOrderCompositions:
    """Complex compositions of transforms."""

    def test_vmap_vjp_jvp(self):
        """vmap over vjp(jvp(...)) - complex nesting."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, seed=1)
        x_nb = tensor_from_jax(x_jax)

        def complex_fn_nb(x):
            # First apply jvp
            t = nb.ones_like(x)
            out1, tan1 = nb.jvp(lambda x: nb.reduce_sum(nb.exp(x)), (x,), (t,))
            # Then vjp on a different function
            out2, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.sin(x)), x)
            (g,) = vjp_fn(nb.ones_like(out2))
            return nb.add(tan1, nb.reduce_sum(g))

        def complex_fn_jax(x):
            t = jnp.ones_like(x)
            out1, tan1 = jax.jvp(lambda x: jnp.sum(jnp.exp(x)), (x,), (t,))
            out2, vjp_fn = jax.vjp(lambda x: jnp.sum(jnp.sin(x)), x)
            (g,) = vjp_fn(jnp.ones_like(out2))
            return tan1 + jnp.sum(g)

        nb_res = nb.vmap(complex_fn_nb)(x_nb)
        jax_res = jax.vmap(complex_fn_jax)(x_jax)

        _close(nb_res, jax_res)

    def test_nested_vmap_vjp(self):
        """vmap(vmap(vjp(...)))."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        def double_vjp_nb(x):
            out, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.mul(x, x)), x)
            (g,) = vjp_fn(nb.ones_like(out))
            return g

        def double_vjp_jax(x):
            out, vjp_fn = jax.vjp(lambda x: jnp.sum(x * x), x)
            (g,) = vjp_fn(jnp.ones_like(out))
            return g

        nb_res = nb.vmap(nb.vmap(double_vjp_nb))(x_nb)
        jax_res = jax.vmap(jax.vmap(double_vjp_jax))(x_jax)

        _close(nb_res, jax_res)
