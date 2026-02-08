# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comprehensive JVP tests across all operation types."""

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


class TestJVPUnary:
    """JVP for unary operations."""

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
    def test_jvp_unary_ops(self, op_nb, op_jax):
        """Test JVP correctness for various unary ops."""
        cleanup_caches()
        x_jax = make_positive_jax_array(3, 4, seed=42)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(op_nb, (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(op_jax, (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_square(self):
        """f(x) = x², tangent direction."""
        cleanup_caches()
        x_jax = make_jax_array(4, 5, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(lambda x: nb.mul(x, x), (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(lambda x: x * x, (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  BINARY OPS
# ═════════════════════════════════════════════════════════════════════════════


class TestJVPBinary:
    """JVP for binary operations."""

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
    def test_jvp_binary_ops(self, op_nb, op_jax):
        """Test JVP for basic binary ops."""
        cleanup_caches()
        x_jax = make_positive_jax_array(3, 4, seed=1)
        y_jax = make_positive_jax_array(3, 4, seed=2)
        tx_jax = jnp.ones_like(x_jax)
        ty_jax = jnp.ones_like(y_jax)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)
        tx_nb = tensor_from_jax(tx_jax)
        ty_nb = tensor_from_jax(ty_jax)

        out_nb, tan_nb = nb.jvp(op_nb, (x_nb, y_nb), (tx_nb, ty_nb))
        out_jax, tan_jax = jax.jvp(op_jax, (x_jax, y_jax), (tx_jax, ty_jax))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_matmul(self):
        """Matrix multiplication."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        y_jax = make_jax_array(4, 5, seed=2)
        tx_jax = jnp.ones_like(x_jax)
        ty_jax = jnp.ones_like(y_jax)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)
        tx_nb = tensor_from_jax(tx_jax)
        ty_nb = tensor_from_jax(ty_jax)

        out_nb, tan_nb = nb.jvp(nb.matmul, (x_nb, y_nb), (tx_nb, ty_nb))
        out_jax, tan_jax = jax.jvp(jnp.matmul, (x_jax, y_jax), (tx_jax, ty_jax))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_broadcast(self):
        """Binary op with broadcasting."""
        cleanup_caches()
        x_jax = make_jax_array(3, 1, seed=1)
        y_jax = make_jax_array(1, 4, seed=2)
        tx_jax = jnp.ones_like(x_jax)
        ty_jax = jnp.ones_like(y_jax)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)
        tx_nb = tensor_from_jax(tx_jax)
        ty_nb = tensor_from_jax(ty_jax)

        out_nb, tan_nb = nb.jvp(nb.add, (x_nb, y_nb), (tx_nb, ty_nb))
        out_jax, tan_jax = jax.jvp(jnp.add, (x_jax, y_jax), (tx_jax, ty_jax))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_partial_tangent(self):
        """Only one input has nonzero tangent."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        y_jax = make_jax_array(3, 4, seed=2)
        tx_jax = jnp.ones_like(x_jax)
        ty_jax = jnp.zeros_like(y_jax)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)
        tx_nb = tensor_from_jax(tx_jax)
        ty_nb = tensor_from_jax(ty_jax)

        out_nb, tan_nb = nb.jvp(nb.mul, (x_nb, y_nb), (tx_nb, ty_nb))
        out_jax, tan_jax = jax.jvp(lambda x, y: x * y, (x_jax, y_jax), (tx_jax, ty_jax))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  REDUCTION OPS
# ═════════════════════════════════════════════════════════════════════════════


class TestJVPReduction:
    """JVP for reduction operations."""

    def test_jvp_reduce_sum_full(self):
        """reduce_sum over all axes."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(nb.reduce_sum, (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(jnp.sum, (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_jvp_reduce_sum_axis(self, axis):
        """reduce_sum along specific axis."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, 5, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(lambda x: nb.reduce_sum(x, axis=axis), (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(lambda x: jnp.sum(x, axis=axis), (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_cumsum(self):
        """Cumulative sum."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(lambda x: nb.cumsum(x, axis=0), (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(lambda x: jnp.cumsum(x, axis=0), (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW OPS
# ═════════════════════════════════════════════════════════════════════════════


class TestJVPView:
    """JVP for view/reshape operations."""

    def test_jvp_reshape(self):
        """Reshape."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(lambda x: nb.reshape(x, (6, 4)), (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(lambda x: jnp.reshape(x, (6, 4)), (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_swap_axes(self):
        """swap_axes (transpose with axes)."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(lambda x: nb.swap_axes(x, 0, 2), (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(lambda x: jnp.swapaxes(x, 0, 2), (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_broadcast_to(self):
        """broadcast_to."""
        cleanup_caches()
        x_jax = make_jax_array(1, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(lambda x: nb.broadcast_to(x, (3, 4)), (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(
            lambda x: jnp.broadcast_to(x, (3, 4)), (x_jax,), (t_jax,)
        )

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_squeeze_unsqueeze(self):
        """squeeze and unsqueeze."""
        cleanup_caches()
        x_jax = make_jax_array(3, 1, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(
            lambda x: nb.unsqueeze(nb.squeeze(x, 1), 1), (x_nb,), (t_nb,)
        )
        out_jax, tan_jax = jax.jvp(
            lambda x: jnp.expand_dims(jnp.squeeze(x, 1), 1), (x_jax,), (t_jax,)
        )

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  COMPOSITIONS
# ═════════════════════════════════════════════════════════════════════════════


class TestJVPComposition:
    """JVP for composed operations."""

    def test_jvp_mlp_layer(self):
        """MLP layer: relu(x @ w + b)."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, seed=1)
        w_jax = make_jax_array(3, 4, seed=2)
        b_jax = make_jax_array(4, seed=3)
        tx_jax = jnp.ones_like(x_jax)
        tw_jax = jnp.ones_like(w_jax)
        tb_jax = jnp.ones_like(b_jax)
        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)
        b_nb = tensor_from_jax(b_jax)
        tx_nb = tensor_from_jax(tx_jax)
        tw_nb = tensor_from_jax(tw_jax)
        tb_nb = tensor_from_jax(tb_jax)

        def f_nb(x, w, b):
            return nb.relu(nb.add(nb.matmul(x, w), b))

        def f_jax(x, w, b):
            return jax.nn.relu(x @ w + b)

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb, w_nb, b_nb), (tx_nb, tw_nb, tb_nb))
        out_jax, tan_jax = jax.jvp(
            f_jax, (x_jax, w_jax, b_jax), (tx_jax, tw_jax, tb_jax)
        )

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_sigmoid_matmul(self):
        """sigmoid(x @ w)."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, seed=1)
        w_jax = make_jax_array(3, 2, seed=2)
        tx_jax = jnp.ones_like(x_jax)
        tw_jax = jnp.ones_like(w_jax)
        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)
        tx_nb = tensor_from_jax(tx_jax)
        tw_nb = tensor_from_jax(tw_jax)

        out_nb, tan_nb = nb.jvp(
            lambda x, w: nb.sigmoid(nb.matmul(x, w)), (x_nb, w_nb), (tx_nb, tw_nb)
        )
        out_jax, tan_jax = jax.jvp(
            lambda x, w: jax.nn.sigmoid(x @ w), (x_jax, w_jax), (tx_jax, tw_jax)
        )

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_nested_ops(self):
        """Deeply nested: exp(sin(log(x + 1)))."""
        cleanup_caches()
        x_jax = make_positive_jax_array(3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        def f_nb(x):
            return nb.exp(nb.sin(nb.log(nb.add(x, 1.0))))

        def f_jax(x):
            return jnp.exp(jnp.sin(jnp.log(x + 1.0)))

        out_nb, tan_nb = nb.jvp(f_nb, (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(f_jax, (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  SPECIAL CASES
# ═════════════════════════════════════════════════════════════════════════════


class TestJVPSpecial:
    """JVP special cases and edge cases."""

    def test_jvp_has_aux(self):
        """has_aux=True passes auxiliary data through."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        t_jax = jnp.ones_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        def f_nb(x):
            y = nb.mul(x, x)
            aux = nb.mean(x)
            return y, aux

        def f_jax(x):
            return x * x, jnp.mean(x)

        out_nb, tan_nb, aux_nb = nb.jvp(f_nb, (x_nb,), (t_nb,), has_aux=True)
        out_jax, tan_jax, aux_jax = jax.jvp(f_jax, (x_jax,), (t_jax,), has_aux=True)

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)
        _close(aux_nb, aux_jax)

    def test_jvp_scalar(self):
        """Scalar input and output."""
        cleanup_caches()
        x_jax = jnp.array(3.0)
        t_jax = jnp.array(1.0)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(lambda x: nb.mul(x, x), (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(lambda x: x * x, (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_zero_tangent(self):
        """Zero tangent should give zero output tangent."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=1)
        t_jax = jnp.zeros_like(x_jax)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        out_nb, tan_nb = nb.jvp(nb.exp, (x_nb,), (t_nb,))
        out_jax, tan_jax = jax.jvp(jnp.exp, (x_jax,), (t_jax,))

        _close(out_nb, out_jax)
        _close(tan_nb, tan_jax)

    def test_jvp_vjp_consistency(self):
        """For scalar f, jvp and vjp should be consistent."""
        cleanup_caches()
        x_jax = jnp.array(2.0)
        x_nb = tensor_from_jax(x_jax)

        def f(x):
            return nb.exp(nb.sin(x))

        # JVP with tangent=1
        _, tan = nb.jvp(f, (x_nb,), (tensor_from_jax(jnp.array(1.0)),))

        # VJP with cotangent=1
        _, vjp_fn = nb.vjp(f, x_nb)
        (grad,) = vjp_fn(tensor_from_jax(jnp.array(1.0)))

        # Should be equal for scalar functions
        _close(tan, to_jax(grad))
