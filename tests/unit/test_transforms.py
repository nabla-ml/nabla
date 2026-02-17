# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for vjp and jvp transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from tests.unit.common import (
    cleanup_caches,
    make_jax_array,
    tensor_from_jax,
    to_jax,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _close(nb_val, jax_val, rtol=5e-4, atol=5e-4):
    """Assert nabla Tensor ≈ JAX array."""
    np.testing.assert_allclose(
        to_jax(nb_val),
        jax_val,
        rtol=rtol,
        atol=atol,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  VJP TESTS
# ═════════════════════════════════════════════════════════════════════════════


class TestVJPBasic:
    """Basic VJP correctness — scalar and vector functions."""

    def test_vjp_scalar_square(self):
        """f(x) = x², vjp with cotangent=1 → gradient = 2x."""
        cleanup_caches()

        x_jax = jnp.array(3.0)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.mul(x, x)

        def f_jax(x):
            return x * x

        # Nabla
        y_nb, vjp_fn = nb.vjp(f_nb, x_nb)
        cot = tensor_from_jax(jnp.array(1.0))
        (g_nb,) = vjp_fn(cot)

        # JAX
        y_jax, vjp_fn_jax = jax.vjp(f_jax, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.array(1.0))

        _close(y_nb, y_jax)
        _close(g_nb, g_jax)

    def test_vjp_two_args(self):
        """f(x, y) = x * y, vjp → (y, x)."""
        cleanup_caches()

        x_jax, y_jax = jnp.array(3.0), jnp.array(4.0)
        x_nb, y_nb = tensor_from_jax(x_jax), tensor_from_jax(y_jax)

        out_nb, vjp_fn = nb.vjp(lambda x, y: nb.mul(x, y), x_nb, y_nb)
        cot = tensor_from_jax(jnp.array(1.0))
        gx_nb, gy_nb = vjp_fn(cot)

        out_jax, vjp_fn_jax = jax.vjp(lambda x, y: x * y, x_jax, y_jax)
        gx_jax, gy_jax = vjp_fn_jax(jnp.array(1.0))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gy_nb, gy_jax)

    def test_vjp_vector(self):
        """Vector function: f(x) = sin(x), vjp."""
        cleanup_caches()

        x_jax = make_jax_array(4)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(nb.sin, x_nb)
        cot = tensor_from_jax(jnp.ones(4))
        (g_nb,) = vjp_fn(cot)

        out_jax, vjp_fn_jax = jax.vjp(jnp.sin, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones(4))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_composition(self):
        """Composed ops: f(x) = sigmoid(matmul(x, w))."""
        cleanup_caches()

        x_jax = make_jax_array(2, 3, seed=1)
        w_jax = make_jax_array(3, 2, seed=2)
        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)

        def f_nb(x, w):
            return nb.sigmoid(nb.matmul(x, w))

        def f_jax(x, w):
            return jax.nn.sigmoid(x @ w)

        out_nb, vjp_fn = nb.vjp(f_nb, x_nb, w_nb)
        cot = tensor_from_jax(jnp.ones_like(jax.nn.sigmoid(x_jax @ w_jax)))
        gx_nb, gw_nb = vjp_fn(cot)

        out_jax, vjp_fn_jax = jax.vjp(f_jax, x_jax, w_jax)
        gx_jax, gw_jax = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gw_nb, gw_jax)

    def test_vjp_has_aux(self):
        """has_aux=True passes auxiliary data through."""
        cleanup_caches()

        x_jax = jnp.array(3.0)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.mul(x, x)
            return y, nb.add(x, x)  # aux = 2*x

        def f_jax(x):
            return x * x, x + x

        out_nb, vjp_fn, aux_nb = nb.vjp(f_nb, x_nb, has_aux=True)
        cot = tensor_from_jax(jnp.array(1.0))
        (g_nb,) = vjp_fn(cot)

        out_jax, vjp_fn_jax, aux_jax = jax.vjp(f_jax, x_jax, has_aux=True)
        (g_jax,) = vjp_fn_jax(jnp.array(1.0))

        _close(out_nb, out_jax)
        _close(aux_nb, aux_jax)
        _close(g_nb, g_jax)


class TestVJPOps:
    """VJP correctness for various ops."""

    @pytest.mark.parametrize(
        "op_nb,op_jax",
        [
            (nb.exp, jnp.exp),
            (nb.log, jnp.log),
            (nb.tanh, jnp.tanh),
            (nb.sin, jnp.sin),
            (nb.cos, jnp.cos),
            (nb.relu, jax.nn.relu),
            (nb.sigmoid, jax.nn.sigmoid),
            (nb.neg, jnp.negative),
            (nb.abs, jnp.abs),
            (nb.sqrt, jnp.sqrt),
        ],
    )
    def test_vjp_unary_ops(self, op_nb, op_jax):
        cleanup_caches()
        # Use positive values to avoid domain issues with log/sqrt
        x_jax = jnp.abs(make_jax_array(4)) + 0.5
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(op_nb, x_nb)
        cot = tensor_from_jax(jnp.ones(4))
        (g_nb,) = vjp_fn(cot)

        out_jax, vjp_fn_jax = jax.vjp(op_jax, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones(4))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  JVP TESTS
# ═════════════════════════════════════════════════════════════════════════════


class TestJVPBasic:
    """Basic JVP correctness — scalar and vector functions."""

    def test_jvp_scalar_square(self):
        """f(x) = x², jvp with tangent=1 → 2x."""
        cleanup_caches()

        x_jax = jnp.array(3.0)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.mul(x, x)

        def f_jax(x):
            return x * x

        t_nb = tensor_from_jax(jnp.array(1.0))

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(f_jax, (x_jax,), (jnp.array(1.0),))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_two_args(self):
        """f(x, y) = x * y, jvp with (1, 0) → y."""
        cleanup_caches()

        x_jax, y_jax = jnp.array(3.0), jnp.array(4.0)
        x_nb, y_nb = tensor_from_jax(x_jax), tensor_from_jax(y_jax)

        def f_nb(x, y):
            return nb.mul(x, y)

        def f_jax(x, y):
            return x * y

        tx = tensor_from_jax(jnp.array(1.0))
        ty = tensor_from_jax(jnp.array(0.0))

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb, y_nb), (tx, ty))
        out_jax, tan_jax = jax.jvp(
            f_jax, (x_jax, y_jax), (jnp.array(1.0), jnp.array(0.0))
        )

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_vector(self):
        """Vector function: f(x) = sin(x)."""
        cleanup_caches()

        x_jax = make_jax_array(4)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(jnp.ones(4))

        out_nb, tan_nb = nb.jvp(nb.sin, (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(jnp.sin, (x_jax,), (jnp.ones(4),))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_composition(self):
        """Composed: f(x) = exp(sin(x))."""
        cleanup_caches()

        x_jax = make_jax_array(4)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(jnp.ones(4))

        def f_nb(x):
            return nb.exp(nb.sin(x))

        def f_jax(x):
            return jnp.exp(jnp.sin(x))

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(f_jax, (x_jax,), (jnp.ones(4),))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    @pytest.mark.parametrize(
        "op_nb,op_jax",
        [
            (nb.exp, jnp.exp),
            (nb.log, jnp.log),
            (nb.tanh, jnp.tanh),
            (nb.sin, jnp.sin),
            (nb.cos, jnp.cos),
            (nb.relu, jax.nn.relu),
            (nb.sigmoid, jax.nn.sigmoid),
            (nb.neg, jnp.negative),
            (nb.sqrt, jnp.sqrt),
        ],
    )
    def test_jvp_unary_ops(self, op_nb, op_jax):
        cleanup_caches()
        x_jax = jnp.abs(make_jax_array(4)) + 0.5
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(jnp.ones(4))

        out_nb, tan_nb = nb.jvp(op_nb, (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(op_jax, (x_jax,), (jnp.ones(4),))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_has_aux(self):
        """has_aux=True passes auxiliary data through."""
        cleanup_caches()

        x_jax = jnp.array(3.0)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.mul(x, x)
            return y, nb.add(x, x)

        def f_jax(x):
            return x * x, x + x

        t_nb = tensor_from_jax(jnp.array(1.0))
        out_nb, tan_nb, aux_nb = nb.jvp(f_nb, (x_nb,), (t_nb,), has_aux=True)
        out_jax, tan_jax, aux_jax = jax.jvp(
            f_jax, (x_jax,), (jnp.array(1.0),), has_aux=True
        )

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)
        _close(aux_nb, aux_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  VJP + JVP consistency: they should give same Jacobian info
# ═════════════════════════════════════════════════════════════════════════════


class TestVJPJVPConsistency:
    def test_vjp_jvp_agree_scalar(self):
        """For scalar f, vjp and jvp should produce consistent results."""
        cleanup_caches()

        x_jax = jnp.array(2.0)
        x_nb = tensor_from_jax(x_jax)

        def f(x):
            return nb.exp(nb.sin(x))

        # JVP: directional derivative with tangent=1
        _, tan = nb.jvp(f, (x_nb,), (tensor_from_jax(jnp.array(1.0)),))

        # VJP: gradient with cotangent=1
        _, vjp_fn = nb.vjp(f, x_nb)
        (grad,) = vjp_fn(tensor_from_jax(jnp.array(1.0)))

        # For scalar → scalar, these should be equal
        _close(tan, to_jax(grad))
