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


def _close(nb_val, jax_val, rtol=5e-4, atol=5e-4):
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
            out, tan = nb.jvp(
                lambda x, y: nb.reduce_sum(nb.mul(x, y)), (x, y), (tx, ty)
            )
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


class TestNestedJacobians:
    """Nested jacrev/jacfwd regression tests."""

    def test_jacfwd_jacrev_hessian_matches_jax(self):
        """jacfwd(jacrev) should match JAX on a stable cubic scalar function."""
        cleanup_caches()
        x_jax = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.mul(nb.mul(x, x), x))

        def f_jax(x):
            return jnp.sum(x * x * x)

        h_nb = nb.jacfwd(nb.jacrev(f_nb))(x_nb)
        h_jax = jax.jacfwd(jax.jacrev(f_jax))(x_jax)
        _close(h_nb, h_jax)

    def test_jacrev_jacfwd_hessian_matches_jax(self):
        """jacrev(jacfwd) should match JAX on the same cubic scalar function."""
        cleanup_caches()
        x_jax = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.mul(nb.mul(x, x), x))

        def f_jax(x):
            return jnp.sum(x * x * x)

        h_nb = nb.jacrev(nb.jacfwd(f_nb))(x_nb)
        h_jax = jax.jacrev(jax.jacfwd(f_jax))(x_jax)
        _close(h_nb, h_jax)

    def test_nested_hessian_forms_are_consistent(self):
        """All supported nested Hessian constructions should agree with each other."""
        cleanup_caches()
        x_jax = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.mul(nb.mul(x, x), x))

        h_fwd_rev = nb.jacfwd(nb.jacrev(f_nb))(x_nb)
        h_rev_fwd = nb.jacrev(nb.jacfwd(f_nb))(x_nb)
        h_rev_rev = nb.jacrev(nb.jacrev(f_nb))(x_nb)

        _close(h_fwd_rev, to_jax(h_rev_fwd))
        _close(h_fwd_rev, to_jax(h_rev_rev))


class TestJaxParityReliabilityMatrix:
    """Reliability matrix against JAX across diverse transform circumstances."""

    @pytest.mark.parametrize(
        "shape,seed",
        [
            ((3,), 11),
            ((2, 3), 13),
        ],
    )
    def test_grad_scalarized_nonlinear_matches_jax(self, shape, seed):
        """grad parity for scalarized nonlinear objectives over multiple shapes."""
        cleanup_caches()
        x_jax = make_positive_jax_array(*shape, seed=seed)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.exp(nb.sin(x)))

        def f_jax(x):
            return jnp.sum(jnp.exp(jnp.sin(x)))

        g_nb = nb.grad(f_nb)(x_nb)
        g_jax = jax.grad(f_jax)(x_jax)
        _close(g_nb, g_jax)

    @pytest.mark.parametrize(
        "shape,seed",
        [
            ((4,), 17),
            ((2, 3), 19),
        ],
    )
    def test_jvp_directional_matches_jax(self, shape, seed):
        """jvp directional derivatives match JAX across varied input ranks."""
        cleanup_caches()
        x_jax = make_jax_array(*shape, seed=seed)
        t_jax = make_jax_array(*shape, seed=seed + 100)
        x_nb = tensor_from_jax(x_jax)
        t_nb = tensor_from_jax(t_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.mul(nb.exp(x), nb.sin(x)))

        def f_jax(x):
            return jnp.sum(jnp.exp(x) * jnp.sin(x))

        _, tan_nb = nb.jvp(f_nb, (x_nb,), (t_nb,))
        _, tan_jax = jax.jvp(f_jax, (x_jax,), (t_jax,))
        _close(tan_nb, tan_jax)

    def test_vjp_binary_scalarized_matches_jax(self):
        """vjp parity for scalarized bilinear objective with two inputs."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=23)
        y_jax = make_jax_array(3, 4, seed=29)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        def f_nb(x, y):
            return nb.reduce_sum(nb.mul(x, y))

        def f_jax(x, y):
            return jnp.sum(x * y)

        out_nb, pull_nb = nb.vjp(f_nb, x_nb, y_nb)
        gx_nb, gy_nb = pull_nb(nb.ones_like(out_nb))

        out_jax, pull_jax = jax.vjp(f_jax, x_jax, y_jax)
        gx_jax, gy_jax = pull_jax(jnp.ones_like(out_jax))

        _close(gx_nb, gx_jax)
        _close(gy_nb, gy_jax)

    def test_jacrev_vector_output_matches_jax(self):
        """jacrev parity for vector-valued function with mixed unary/binary ops."""
        cleanup_caches()
        x_jax = make_jax_array(5, seed=31)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.add(nb.sin(x), nb.mul(x, x))

        def f_jax(x):
            return jnp.sin(x) + x * x

        j_nb = nb.jacrev(f_nb)(x_nb)
        j_jax = jax.jacrev(f_jax)(x_jax)
        _close(j_nb, j_jax)

    def test_vmapped_jacrev_matches_jax(self):
        """vmap(jacrev) parity for per-example Jacobians."""
        cleanup_caches()
        x_jax = make_jax_array(4, 3, seed=37)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.add(nb.sin(x), nb.mul(x, x))

        def f_jax(x):
            return jnp.sin(x) + x * x

        j_nb = nb.vmap(nb.jacrev(f_nb))(x_nb)
        j_jax = jax.vmap(jax.jacrev(f_jax))(x_jax)
        _close(j_nb, j_jax)

    def test_vmapped_jacfwd_matches_jax(self):
        """vmap(jacfwd) parity for per-example Jacobians."""
        cleanup_caches()
        x_jax = make_jax_array(4, 3, seed=41)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.add(nb.sin(x), nb.mul(x, x))

        def f_jax(x):
            return jnp.sin(x) + x * x

        j_nb = nb.vmap(nb.jacfwd(f_nb))(x_nb)
        j_jax = jax.vmap(jax.jacfwd(f_jax))(x_jax)
        _close(j_nb, j_jax)

    def test_nested_hessian_view_reduce_chain_matches_jax(self):
        """Nested Hessian parity on explicit logical view+reduce chain."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, seed=47)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.reshape(x, (3, 2))
            y = nb.unsqueeze(y, axis=0)
            y = nb.moveaxis(y, source=0, destination=2)
            y = nb.reduce_sum(nb.mul(nb.sin(y), y), axis=-1, keepdims=False)
            return nb.reduce_sum(y)

        def f_jax(x):
            y = jnp.reshape(x, (3, 2))
            y = jnp.expand_dims(y, axis=0)
            y = jnp.moveaxis(y, 0, 2)
            y = jnp.sum(jnp.sin(y) * y, axis=-1, keepdims=False)
            return jnp.sum(y)

        h_nb = nb.jacfwd(nb.jacrev(f_nb))(x_nb)
        h_jax = jax.jacfwd(jax.jacrev(f_jax))(x_jax)
        _close(h_nb, h_jax)

    def test_vmapped_jacrev_view_chain_matches_jax(self):
        """vmap(jacrev) parity for vector-output function with view/reduce chain."""
        cleanup_caches()
        x_jax = make_jax_array(4, 2, 3, seed=53)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.swap_axes(x, axis1=0, axis2=1)
            y = nb.unsqueeze(y, axis=0)
            y = nb.squeeze(y, axis=0)
            return nb.reduce_sum(nb.mul(y, y), axis=1, keepdims=False)

        def f_jax(x):
            y = jnp.swapaxes(x, 0, 1)
            y = jnp.expand_dims(y, axis=0)
            y = jnp.squeeze(y, axis=0)
            return jnp.sum(y * y, axis=1, keepdims=False)

        j_nb = nb.vmap(nb.jacrev(f_nb))(x_nb)
        j_jax = jax.vmap(jax.jacrev(f_jax))(x_jax)
        _close(j_nb, j_jax)


# ═════════════════════════════════════════════════════════════════════════════
#  NESTED HIGHER-ORDER TRANSFORMS ON REDUCE AND VIEW OPS
# ═════════════════════════════════════════════════════════════════════════════


class TestNestedHigherOrderOnReduceViewOps:
    """Hessians (nested jacrev/jacfwd) applied individually to each reduce and
    view op, validated against JAX.  Covers:
      - reduce_sum (global and axis)
      - mean (axis)
      - reshape, broadcast_to, moveaxis, swap_axes, unsqueeze, squeeze
      - a combined view+reduce chain
    All tested under three Hessian constructions:
      fwd_rev  = jacfwd(jacrev(f))
      rev_fwd  = jacrev(jacfwd(f))
      rev_rev  = jacrev(jacrev(f))
    """

    @staticmethod
    def _make_hessian_nb(method, f_nb):
        if method == "fwd_rev":
            return nb.jacfwd(nb.jacrev(f_nb))
        elif method == "rev_fwd":
            return nb.jacrev(nb.jacfwd(f_nb))
        else:  # rev_rev
            return nb.jacrev(nb.jacrev(f_nb))

    @staticmethod
    def _make_hessian_jax(method, f_jax):
        if method == "fwd_rev":
            return jax.jacfwd(jax.jacrev(f_jax))
        elif method == "rev_fwd":
            return jax.jacrev(jax.jacfwd(f_jax))
        else:  # rev_rev
            return jax.jacrev(jax.jacrev(f_jax))

    # ── reduce_sum (global) ────────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_reduce_sum_global(self, method):
        """Hessian of sum(x^3): expected diag(6x), tests reduce_sum globally."""
        cleanup_caches()
        x_jax = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.mul(nb.mul(x, x), x))

        def f_jax(x):
            return jnp.sum(x * x * x)

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )

    # ── reduce_sum over axis ───────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_reduce_sum_axis(self, method):
        """Hessian of sum(sin(sum_axis0(x^2))), x:(3,4), tests axis reduce."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=7)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.sin(nb.reduce_sum(nb.mul(x, x), axis=0)))

        def f_jax(x):
            return jnp.sum(jnp.sin(jnp.sum(x * x, axis=0)))

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )

    # ── mean ──────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_mean(self, method):
        """Hessian of sum(exp(mean(x^2, axis=-1))), x:(3,4), tests mean."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=11)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            return nb.reduce_sum(nb.exp(nb.mean(nb.mul(x, x), axis=-1)))

        def f_jax(x):
            return jnp.sum(jnp.exp(jnp.mean(x * x, axis=-1)))

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )

    # ── reshape ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_reshape(self, method):
        """Hessian of sum((reshape(x,(12,)))^3), x:(3,4), tests reshape."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=13)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.reshape(x, (12,))
            return nb.reduce_sum(nb.mul(nb.mul(y, y), y))

        def f_jax(x):
            y = jnp.reshape(x, (12,))
            return jnp.sum(y * y * y)

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )

    # ── broadcast_to ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_broadcast_to(self, method):
        """Hessian of sum(broadcast_to(sin(x),(2,4))^2), x:(4,), tests broadcast_to."""
        cleanup_caches()
        x_jax = make_jax_array(4, seed=17)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.broadcast_to(nb.sin(x), (2, 4))
            return nb.reduce_sum(nb.mul(y, y))

        def f_jax(x):
            y = jnp.broadcast_to(jnp.sin(x), (2, 4))
            return jnp.sum(y * y)

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )

    # ── moveaxis ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_moveaxis(self, method):
        """Hessian of sum(moveaxis(x,0,1)^3), x:(3,4), tests moveaxis."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=19)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.moveaxis(x, source=0, destination=1)
            return nb.reduce_sum(nb.mul(nb.mul(y, y), y))

        def f_jax(x):
            y = jnp.moveaxis(x, 0, 1)
            return jnp.sum(y * y * y)

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )

    # ── swap_axes ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_swap_axes(self, method):
        """Hessian of sum(sin(swap_axes(x,0,1))*swap_axes(x,0,1)), x:(3,4)."""
        cleanup_caches()
        x_jax = make_jax_array(3, 4, seed=23)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.swap_axes(x, axis1=0, axis2=1)
            return nb.reduce_sum(nb.mul(nb.sin(y), y))

        def f_jax(x):
            y = jnp.swapaxes(x, 0, 1)
            return jnp.sum(jnp.sin(y) * y)

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )

    # ── unsqueeze + squeeze ───────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_unsqueeze_squeeze(self, method):
        """Hessian through an unsqueeze→squeeze round-trip, x:(4,)."""
        cleanup_caches()
        x_jax = make_jax_array(4, seed=29)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.unsqueeze(x, axis=0)   # (1,4)
            y = nb.squeeze(y, axis=0)     # (4,)
            return nb.reduce_sum(nb.mul(nb.mul(y, y), y))

        def f_jax(x):
            y = jnp.expand_dims(x, axis=0)
            y = jnp.squeeze(y, axis=0)
            return jnp.sum(y * y * y)

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )

    # ── combined view + reduce chain ──────────────────────────────────────

    @pytest.mark.parametrize("method", ["fwd_rev", "rev_fwd", "rev_rev"])
    def test_hessian_combined_view_reduce_chain(self, method):
        """Hessian through reshape→unsqueeze→moveaxis→reduce_sum (axis) chain."""
        cleanup_caches()
        x_jax = make_jax_array(2, 3, seed=37)
        x_nb = tensor_from_jax(x_jax)

        def f_nb(x):
            y = nb.reshape(x, (3, 2))
            y = nb.unsqueeze(y, axis=0)                      # (1,3,2)
            y = nb.moveaxis(y, source=0, destination=2)      # (3,2,1)
            y = nb.reduce_sum(nb.mul(nb.sin(y), y), axis=-1) # (3,2)
            return nb.reduce_sum(y)

        def f_jax(x):
            y = jnp.reshape(x, (3, 2))
            y = jnp.expand_dims(y, axis=0)
            y = jnp.moveaxis(y, 0, 2)
            y = jnp.sum(jnp.sin(y) * y, axis=-1)
            return jnp.sum(y)

        _close(
            self._make_hessian_nb(method, f_nb)(x_nb),
            self._make_hessian_jax(method, f_jax)(x_jax),
        )
