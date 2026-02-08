# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comprehensive VJP tests across all operation types."""

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


def _close(nb_val, jax_val, rtol=5e-4, atol=5e-4):
    """Assert nabla Tensor ≈ JAX array."""
    np.testing.assert_allclose(to_jax(nb_val), jax_val, rtol=rtol, atol=atol)


# ═════════════════════════════════════════════════════════════════════════════
#  UNARY OPS
# ═════════════════════════════════════════════════════════════════════════════


class TestVJPUnary:
    """VJP for unary operations."""

    @pytest.mark.parametrize(
        "op_nb,op_jax",
        [
            (nb.exp, jnp.exp),
            (nb.log, jnp.log),
            (nb.sin, jnp.sin),
            (nb.cos, jnp.cos),
            (nb.tanh, jnp.tanh),
            (nb.sqrt, jnp.sqrt),
            (nb.neg, jnp.negative),
            (nb.abs, jnp.abs),
            (nb.relu, jax.nn.relu),
            (nb.sigmoid, jax.nn.sigmoid),
            (nb.gelu, jax.nn.gelu),
        ],
    )
    def test_vjp_unary_ops(self, op_nb, op_jax):
        """Test VJP correctness for various unary ops."""
        cleanup_caches()
        x_jax = make_positive_jax_array(3, 4, seed=42)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(op_nb, x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(op_jax, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_square(self):
        """f(x) = x²."""
        cleanup_caches()
        x_jax = make_jax_array(4, 5, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.mul(x, x), x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: x * x, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  BINARY OPS
# ═════════════════════════════════════════════════════════════════════════════


class TestVJPBinary:
    """VJP for binary operations."""

    @pytest.mark.parametrize(
        "op_nb,op_jax",
        [
            (nb.add, lambda x, y: x + y),
            (nb.sub, lambda x, y: x - y),
            (nb.mul, lambda x, y: x * y),
            (nb.div, lambda x, y: x / y),
            (nb.pow, lambda x, y: x**y),
        ],
    )
    def test_vjp_binary_ops(self, op_nb, op_jax):
        """Test VJP for basic binary ops."""
        cleanup_caches()
        x_jax = make_positive_jax_array(3, 4, seed=1)
        y_jax = make_positive_jax_array(3, 4, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        out_nb, vjp_fn = nb.vjp(op_nb, x_nb, y_nb)
        gx_nb, gy_nb = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(op_jax, x_jax, y_jax)
        gx_jax, gy_jax = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gy_nb, gy_jax)

    def test_vjp_matmul(self):
        """Matrix multiplication."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        y_jax = make_jax_array(4, 5, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        out_nb, vjp_fn = nb.vjp(nb.matmul, x_nb, y_nb)
        gx_nb, gy_nb = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(jnp.matmul, x_jax, y_jax)
        gx_jax, gy_jax = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gy_nb, gy_jax)

    def test_vjp_broadcast(self):
        """Binary op with broadcasting."""
        cleanup_caches()
        x_jax = make_jax_array(3, 1, seed=1)
        y_jax = make_jax_array(1, 4, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        out_nb, vjp_fn = nb.vjp(nb.add, x_nb, y_nb)
        gx_nb, gy_nb = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(jnp.add, x_jax, y_jax)
        gx_jax, gy_jax = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gy_nb, gy_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  REDUCTION OPS
# ═════════════════════════════════════════════════════════════════════════════


class TestVJPReduction:
    """VJP for reduction operations."""

    def test_vjp_reduce_sum_full(self):
        """reduce_sum over all axes."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(nb.reduce_sum, x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(jnp.sum, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_vjp_reduce_sum_axis(self, axis):
        """reduce_sum along specific axis."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, 5, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(x, axis=axis), x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(x, axis=axis), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_reduce_mean(self):
        """reduce_mean."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.mean(x)), x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(jnp.mean(x)), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_reduce_max(self):
        """reduce_max."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.reduce_max(x)), x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(jnp.max(x)), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_softmax(self):
        """Softmax."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.softmax(x)), x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(jax.nn.softmax(x)), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW OPS
# ═════════════════════════════════════════════════════════════════════════════


class TestVJPView:
    """VJP for view/reshape operations."""

    def test_vjp_reshape(self):
        """Reshape."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.reshape(x, (6, 4))), x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(jnp.reshape(x, (6, 4))), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_swap_axes(self):
        """swap_axes (transpose with axes)."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.reduce_sum(nb.swap_axes(x, 0, 2)), x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: jnp.sum(jnp.swapaxes(x, 0, 2)), x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_broadcast_to(self):
        """broadcast_to."""
        cleanup_caches()
        x_jax = make_jax_array(1, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(
            lambda x: nb.reduce_sum(nb.broadcast_to(x, (3, 4))), x_nb
        )
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(
            lambda x: jnp.sum(jnp.broadcast_to(x, (3, 4))), x_jax
        )
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_squeeze_unsqueeze(self):
        """squeeze and unsqueeze."""
        cleanup_caches()
        x_jax = make_jax_array(3, 1, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(
            lambda x: nb.reduce_sum(nb.unsqueeze(nb.squeeze(x, 1), 1)), x_nb
        )
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(
            lambda x: jnp.sum(jnp.expand_dims(jnp.squeeze(x, 1), 1)), x_jax
        )
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  COMPOSITIONS
# ═════════════════════════════════════════════════════════════════════════════


class TestVJPComposition:
    """VJP for composed operations."""

    def test_vjp_mlp_layer(self):
        """MLP layer: relu(x @ w + b)."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, seed=1)
        w_jax = make_jax_array(3, 4, seed=2)
        b_jax = make_jax_array(4, seed=3)
        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)
        b_nb = tensor_from_jax(b_jax)

        def f_nb(x, w, b):
            return nb.relu(nb.add(nb.matmul(x, w), b))

        def f_jax(x, w, b):
            return jax.nn.relu(x @ w + b)

        out_nb, vjp_fn = nb.vjp(f_nb, x_nb, w_nb, b_nb)
        gx_nb, gw_nb, gb_nb = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(f_jax, x_jax, w_jax, b_jax)
        gx_jax, gw_jax, gb_jax = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gw_nb, gw_jax)
        _close(gb_nb, gb_jax)

    def test_vjp_sigmoid_matmul(self):
        """sigmoid(x @ w)."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, seed=1)
        w_jax = make_jax_array(3, 2, seed=2)
        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)

        out_nb, vjp_fn = nb.vjp(lambda x, w: nb.sigmoid(nb.matmul(x, w)), x_nb, w_nb)
        gx_nb, gw_nb = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x, w: jax.nn.sigmoid(x @ w), x_jax, w_jax)
        gx_jax, gw_jax = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(gx_nb, gx_jax)
        _close(gw_nb, gw_jax)

    def test_vjp_nested_ops(self):
        """Deeply nested: exp(sin(log(x + 1)))."""
        cleanup_caches()
        x_jax = make_positive_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.exp(nb.sin(nb.log(nb.add(x, 1.0)))))

        def f_jax(x):
            return jnp.sum(jnp.exp(jnp.sin(jnp.log(x + 1.0))))

        out_nb, vjp_fn = nb.vjp(f_nb, x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(f_jax, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  SPECIAL CASES
# ═════════════════════════════════════════════════════════════════════════════


class TestVJPSpecial:
    """VJP special cases and edge cases."""

    def test_vjp_has_aux(self):
        """has_aux=True passes auxiliary data through."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.reduce_sum(nb.mul(x, x))
            aux = nb.mean(x)
            return y, aux

        def f_jax(x):
            return jnp.sum(x * x), jnp.mean(x)

        out_nb, vjp_fn, aux_nb = nb.vjp(f_nb, x_nb, has_aux=True)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax, aux_jax = jax.vjp(f_jax, x_jax, has_aux=True)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(aux_nb, aux_jax)
        _close(g_nb, g_jax)

    def test_vjp_scalar(self):
        """Scalar input and output."""
        cleanup_caches()
        x_jax = jnp.array(3.0)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(lambda x: nb.mul(x, x), x_nb)
        (g_nb,) = vjp_fn(nb.ones_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(lambda x: x * x, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.ones_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)

    def test_vjp_zero_cotangent(self):
        """Zero cotangent should give zero gradient."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        x_nb = tensor_from_jax(x_jax)

        out_nb, vjp_fn = nb.vjp(nb.exp, x_nb)
        (g_nb,) = vjp_fn(nb.zeros_like(out_nb))

        out_jax, vjp_fn_jax = jax.vjp(jnp.exp, x_jax)
        (g_jax,) = vjp_fn_jax(jnp.zeros_like(out_jax))

        _close(out_nb, out_jax)
        _close(g_nb, g_jax)
