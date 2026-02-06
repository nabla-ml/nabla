# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comprehensive tests for jacrev and jacfwd.

Tests verify correctness against JAX's jax.jacrev and jax.jacfwd.
"""

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


def _close(nb_val, jax_val, rtol=1e-3, atol=1e-3):
    """Assert nabla Tensor ≈ JAX array."""
    np.testing.assert_allclose(to_jax(nb_val), jax_val, rtol=rtol, atol=atol)


# ═════════════════════════════════════════════════════════════════════════════
# JACREV TESTS
# ═════════════════════════════════════════════════════════════════════════════


class TestJacrevBasic:
    """Basic jacrev tests: scalar/vector functions."""

    def test_jacrev_scalar_to_scalar(self):
        """f: R -> R, Jacobian is scalar (derivative)."""
        cleanup_caches()

        def f_nb(x):
            return nb.reduce_sum(x * x)  # f(x) = x^2

        def f_jax(x):
            return jnp.sum(x * x)

        x_jax = make_jax_array(1)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_vector_to_scalar(self):
        """f: R^n -> R, Jacobian shape is (n,)."""
        cleanup_caches()

        def f_nb(x):
            return nb.reduce_sum(x * x)

        def f_jax(x):
            return jnp.sum(x * x)

        x_jax = make_jax_array(4)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_vector_to_vector(self):
        """f: R^n -> R^m, Jacobian shape is (m, n)."""
        cleanup_caches()

        def f_nb(x):
            return x * x  # element-wise square, R^n -> R^n

        def f_jax(x):
            return x * x

        x_jax = make_jax_array(3)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_matrix_function(self):
        """f: R^(n,m) -> R, Jacobian shape matches input (n, m)."""
        cleanup_caches()

        def f_nb(x):
            return nb.reduce_sum(x * x)

        def f_jax(x):
            return jnp.sum(x * x)

        x_jax = make_jax_array(3, 4)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_matmul(self):
        """Jacobian of matmul-based linear transform."""
        cleanup_caches()

        w_jax = make_jax_array(3, 2, seed=1)
        w_nb = tensor_from_jax(w_jax)

        def f_nb(x):
            return nb.matmul(x, w_nb)

        def f_jax(x):
            return x @ w_jax

        x_jax = make_jax_array(4, 3, seed=2)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)


class TestJacrevOps:
    """jacrev with various operations."""

    def test_jacrev_exp(self):
        """Jacobian of exp."""
        cleanup_caches()

        def f_nb(x):
            return nb.exp(x)

        def f_jax(x):
            return jnp.exp(x)

        x_jax = make_jax_array(3)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_sin(self):
        """Jacobian of sin."""
        cleanup_caches()

        def f_nb(x):
            return nb.sin(x)

        def f_jax(x):
            return jnp.sin(x)

        x_jax = make_jax_array(4)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_matmul(self):
        """Jacobian of matmul-based linear transform."""
        cleanup_caches()

        w_jax = make_jax_array(3, 2, seed=1)
        w_nb = tensor_from_jax(w_jax)

        def f_nb(x):
            return nb.matmul(x, w_nb)

        def f_jax(x):
            return x @ w_jax

        x_jax = make_jax_array(4, 3, seed=2)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_composite(self):
        """Jacobian of composite function."""
        cleanup_caches()

        def f_nb(x):
            return nb.reduce_sum(nb.sin(x * x))

        def f_jax(x):
            return jnp.sum(jnp.sin(x * x))

        x_jax = make_jax_array(5)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacrev(f_nb)(x_nb)
        jac_jax = jax.jacrev(f_jax)(x_jax)

        _close(jac_nb, jac_jax)


class TestJacrevMultiArg:
    """jacrev with multiple arguments and argnums."""

    def test_jacrev_argnums_single(self):
        """Differentiate w.r.t. a single arg."""
        cleanup_caches()

        def f_nb(x, y):
            return nb.reduce_sum(x * y)

        def f_jax(x, y):
            return jnp.sum(x * y)

        x_jax = make_jax_array(4, seed=1)
        y_jax = make_jax_array(4, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        # argnums=0: differentiate w.r.t. x only
        jac_nb = nb.jacrev(f_nb, argnums=0)(x_nb, y_nb)
        jac_jax = jax.jacrev(f_jax, argnums=0)(x_jax, y_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_argnums_second(self):
        """Differentiate w.r.t. second arg."""
        cleanup_caches()

        def f_nb(x, y):
            return nb.reduce_sum(x * y)

        def f_jax(x, y):
            return jnp.sum(x * y)

        x_jax = make_jax_array(4, seed=1)
        y_jax = make_jax_array(4, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        jac_nb = nb.jacrev(f_nb, argnums=1)(x_nb, y_nb)
        jac_jax = jax.jacrev(f_jax, argnums=1)(x_jax, y_jax)

        _close(jac_nb, jac_jax)

    def test_jacrev_argnums_tuple(self):
        """Differentiate w.r.t. multiple args — returns tuple of Jacobians."""
        cleanup_caches()

        def f_nb(x, y):
            return nb.reduce_sum(x * y)

        def f_jax(x, y):
            return jnp.sum(x * y)

        x_jax = make_jax_array(3, seed=1)
        y_jax = make_jax_array(3, seed=2)
        x_nb = tensor_from_jax(x_jax)
        y_nb = tensor_from_jax(y_jax)

        jac_nb = nb.jacrev(f_nb, argnums=(0, 1))(x_nb, y_nb)
        jac_jax = jax.jacrev(f_jax, argnums=(0, 1))(x_jax, y_jax)

        assert isinstance(jac_nb, tuple) and len(jac_nb) == 2
        _close(jac_nb[0], jac_jax[0])
        _close(jac_nb[1], jac_jax[1])


# ═════════════════════════════════════════════════════════════════════════════
# JACFWD TESTS
# ═════════════════════════════════════════════════════════════════════════════


class TestJacfwdBasic:
    """Basic jacfwd tests: scalar/vector functions."""

    def test_jacfwd_scalar_to_scalar(self):
        """f: R -> R, Jacobian is scalar (derivative)."""
        cleanup_caches()

        def f_nb(x):
            return nb.reduce_sum(x * x)

        def f_jax(x):
            return jnp.sum(x * x)

        x_jax = make_jax_array(1)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacfwd(f_nb)(x_nb)
        jac_jax = jax.jacfwd(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacfwd_vector_to_scalar(self):
        """f: R^n -> R, Jacobian shape is (n,)."""
        cleanup_caches()

        def f_nb(x):
            return nb.reduce_sum(x * x)

        def f_jax(x):
            return jnp.sum(x * x)

        x_jax = make_jax_array(4)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacfwd(f_nb)(x_nb)
        jac_jax = jax.jacfwd(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacfwd_vector_to_vector(self):
        """f: R^n -> R^m, Jacobian shape is (m, n)."""
        cleanup_caches()

        def f_nb(x):
            return x * x

        def f_jax(x):
            return x * x

        x_jax = make_jax_array(3)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacfwd(f_nb)(x_nb)
        jac_jax = jax.jacfwd(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    @pytest.mark.xfail(
        reason="Framework bug: vmap(jvp) + reduce_sum(axis=None) on 2D+ inputs "
               "fails during graph compilation (squeeze shape mismatch)",
        strict=True,
    )
    def test_jacfwd_matrix_function(self):
        """f: R^(n,m) -> R, Jacobian shape matches input (n, m)."""
        cleanup_caches()

        def f_nb(x):
            return nb.reduce_sum(x * x)

        def f_jax(x):
            return jnp.sum(x * x)

        x_jax = make_jax_array(3, 4)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacfwd(f_nb)(x_nb)
        jac_jax = jax.jacfwd(f_jax)(x_jax)

        _close(jac_nb, jac_jax)


class TestJacfwdOps:
    """jacfwd with various operations."""

    def test_jacfwd_exp(self):
        """Jacobian of exp."""
        cleanup_caches()

        def f_nb(x):
            return nb.exp(x)

        def f_jax(x):
            return jnp.exp(x)

        x_jax = make_jax_array(3)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacfwd(f_nb)(x_nb)
        jac_jax = jax.jacfwd(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacfwd_sin(self):
        """Jacobian of sin."""
        cleanup_caches()

        def f_nb(x):
            return nb.sin(x)

        def f_jax(x):
            return jnp.sin(x)

        x_jax = make_jax_array(4)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacfwd(f_nb)(x_nb)
        jac_jax = jax.jacfwd(f_jax)(x_jax)

        _close(jac_nb, jac_jax)

    def test_jacfwd_composite(self):
        """Jacobian of composite function."""
        cleanup_caches()

        def f_nb(x):
            return nb.reduce_sum(nb.sin(x * x))

        def f_jax(x):
            return jnp.sum(jnp.sin(x * x))

        x_jax = make_jax_array(5)
        x_nb = tensor_from_jax(x_jax)

        jac_nb = nb.jacfwd(f_nb)(x_nb)
        jac_jax = jax.jacfwd(f_jax)(x_jax)

        _close(jac_nb, jac_jax)


# ═════════════════════════════════════════════════════════════════════════════
# JACREV vs JACFWD CONSISTENCY
# ═════════════════════════════════════════════════════════════════════════════


class TestJacConsistency:
    """jacrev and jacfwd should produce the same Jacobian."""

    def test_consistency_vector_to_scalar(self):
        """Both transforms agree for f: R^n -> R."""
        cleanup_caches()

        def f(x):
            return nb.reduce_sum(x * x)

        x_jax = make_jax_array(5)
        x = tensor_from_jax(x_jax)

        jac_rev = nb.jacrev(f)(x)
        jac_fwd = nb.jacfwd(f)(x)

        np.testing.assert_allclose(
            to_jax(jac_rev), to_jax(jac_fwd), rtol=1e-3, atol=1e-3
        )

    def test_consistency_vector_to_vector(self):
        """Both transforms agree for f: R^n -> R^n."""
        cleanup_caches()

        def f(x):
            return nb.sin(x) * nb.cos(x)

        x_jax = make_jax_array(4)
        x = tensor_from_jax(x_jax)

        jac_rev = nb.jacrev(f)(x)
        jac_fwd = nb.jacfwd(f)(x)

        np.testing.assert_allclose(
            to_jax(jac_rev), to_jax(jac_fwd), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.xfail(
        reason="Framework bug: vmap(jvp) + reduce_sum(axis=None) on 2D+ inputs",
        strict=True,
    )
    def test_consistency_matrix_input(self):
        """Both transforms agree for f: R^(n,m) -> R."""
        cleanup_caches()

        def f(x):
            return nb.reduce_sum(nb.exp(x))

        x_jax = make_jax_array(2, 3)
        x = tensor_from_jax(x_jax)

        jac_rev = nb.jacrev(f)(x)
        jac_fwd = nb.jacfwd(f)(x)

        np.testing.assert_allclose(
            to_jax(jac_rev), to_jax(jac_fwd), rtol=1e-3, atol=1e-3
        )
