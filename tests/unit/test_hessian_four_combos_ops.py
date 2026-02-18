# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Validate all 4 jacrev/jacfwd Hessian compositions across op chains.

Compositions:
- jacrev(jacrev)
- jacfwd(jacfwd)
- jacrev(jacfwd)
- jacfwd(jacrev)
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from tests.unit.common import cleanup_caches, make_jax_array, tensor_from_jax, to_jax


@pytest.fixture(autouse=True)
def cleanup():
    cleanup_caches()
    yield
    cleanup_caches()


def _hessian_nb(mode: str, f: Callable):
    if mode == "rev_rev":
        return nb.jacrev(nb.jacrev(f))
    if mode == "fwd_fwd":
        return nb.jacfwd(nb.jacfwd(f))
    if mode == "rev_fwd":
        return nb.jacrev(nb.jacfwd(f))
    if mode == "fwd_rev":
        return nb.jacfwd(nb.jacrev(f))
    raise ValueError(mode)


def _hessian_jax(mode: str, f: Callable):
    if mode == "rev_rev":
        return jax.jacrev(jax.jacrev(f))
    if mode == "fwd_fwd":
        return jax.jacfwd(jax.jacfwd(f))
    if mode == "rev_fwd":
        return jax.jacrev(jax.jacfwd(f))
    if mode == "fwd_rev":
        return jax.jacfwd(jax.jacrev(f))
    raise ValueError(mode)


@pytest.mark.parametrize("mode", ["rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev"])
def test_hessian_combo_unary_exp_sum(mode: str):
    x_jax = make_jax_array(5, seed=5)
    x_nb = tensor_from_jax(x_jax)

    f_nb = lambda x: nb.reduce_sum(nb.exp(x))
    f_jax = lambda x: jnp.sum(jnp.exp(x))

    nb_h = _hessian_nb(mode, f_nb)(x_nb)
    jax_h = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(nb_h), np.array(jax_h), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("mode", ["rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev"])
def test_hessian_combo_binary_poly(mode: str):
    x_jax = make_jax_array(4, seed=7)
    x_nb = tensor_from_jax(x_jax)

    f_nb = lambda x: nb.reduce_sum((x * x * x) + (2.0 * x * x) - (5.0 * x) + 3.0)
    f_jax = lambda x: jnp.sum((x * x * x) + (2.0 * x * x) - (5.0 * x) + 3.0)

    nb_h = _hessian_nb(mode, f_nb)(x_nb)
    jax_h = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(nb_h), np.array(jax_h), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("mode", ["rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev"])
def test_hessian_combo_matmul_quadratic(mode: str):
    w_jax = make_jax_array(3, 4, seed=11)
    w_nb = tensor_from_jax(w_jax)
    x_jax = make_jax_array(4, seed=13)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.matmul(w_nb, x)
        return nb.reduce_sum(y * y)

    def f_jax(x):
        y = w_jax @ x
        return jnp.sum(y * y)

    nb_h = _hessian_nb(mode, f_nb)(x_nb)
    jax_h = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(nb_h), np.array(jax_h), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("mode", ["rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev"])
def test_hessian_combo_reduce_axis_chain(mode: str):
    x_jax = make_jax_array(3, 4, seed=17)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.reduce_sum(x * x, axis=0)
        return nb.reduce_sum(nb.sin(y))

    def f_jax(x):
        y = jnp.sum(x * x, axis=0)
        return jnp.sum(jnp.sin(y))

    nb_h = _hessian_nb(mode, f_nb)(x_nb)
    jax_h = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(nb_h), np.array(jax_h), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", ["rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev"])
def test_hessian_combo_view_chain(mode: str):
    x_jax = make_jax_array(2, 3, seed=19)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.reshape(x, (3, 2))
        y = nb.moveaxis(y, 0, 1)
        y = nb.swap_axes(y, 0, 1)
        return nb.reduce_sum(y * y * y)

    def f_jax(x):
        y = jnp.reshape(x, (3, 2))
        y = jnp.moveaxis(y, 0, 1)
        y = jnp.swapaxes(y, 0, 1)
        return jnp.sum(y * y * y)

    nb_h = _hessian_nb(mode, f_nb)(x_nb)
    jax_h = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(nb_h), np.array(jax_h), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", ["rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev"])
def test_hessian_combo_broadcast_chain(mode: str):
    x_jax = make_jax_array(4, seed=23)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.broadcast_to(nb.sin(x), (2, 4))
        return nb.reduce_sum(y * y)

    def f_jax(x):
        y = jnp.broadcast_to(jnp.sin(x), (2, 4))
        return jnp.sum(y * y)

    nb_h = _hessian_nb(mode, f_nb)(x_nb)
    jax_h = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(nb_h), np.array(jax_h), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", ["rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev"])
def test_hessian_combo_getitem_scalar(mode: str):
    x_jax = jnp.array([2.0, 3.0], dtype=jnp.float32)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        return x[0] ** 2 * x[1] + x[1] ** 3

    def f_jax(x):
        return x[0] ** 2 * x[1] + x[1] ** 3

    nb_h = _hessian_nb(mode, f_nb)(x_nb)
    jax_h = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(nb_h), np.array(jax_h), rtol=2e-3, atol=2e-3)
