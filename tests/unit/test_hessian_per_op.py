# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Per-operation Hessian parity tests.

This file follows a strict per-op approach to expose incorrect JVP/VJP rules under
nested transforms, especially jacfwd(grad(.)) and jacrev(grad(.)).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from tests.unit.common import cleanup_caches, make_jax_array, tensor_from_jax, to_jax


def _assert_hessian_close(
    f_nb: Callable,
    f_jax: Callable,
    x_nb,
    x_jax,
    *,
    mode: str,
    rtol: float = 2e-3,
    atol: float = 2e-3,
) -> None:
    if mode == "fwd_grad":
        h_nb = nb.jacfwd(nb.grad(f_nb))(x_nb)
        h_jax = jax.jacfwd(jax.grad(f_jax))(x_jax)
    elif mode == "rev_grad":
        h_nb = nb.jacrev(nb.grad(f_nb))(x_nb)
        h_jax = jax.jacrev(jax.grad(f_jax))(x_jax)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    np.testing.assert_allclose(to_jax(h_nb), np.array(h_jax), rtol=rtol, atol=atol)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
@pytest.mark.parametrize("shape", [(5,), (2, 3)])
def test_unary_exp_hessian(mode: str, shape: tuple[int, ...]):
    cleanup_caches()

    x_jax = make_jax_array(*shape, seed=11)
    x_nb = tensor_from_jax(x_jax)

    f_nb = lambda x: nb.reduce_sum(nb.exp(x))
    f_jax = lambda x: jnp.sum(jnp.exp(x))

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
@pytest.mark.parametrize("shape", [(4,), (2, 3)])
def test_binary_mul_add_hessian(mode: str, shape: tuple[int, ...]):
    cleanup_caches()

    x_jax = make_jax_array(*shape, seed=13)
    x_nb = tensor_from_jax(x_jax)

    f_nb = lambda x: nb.reduce_sum((x * x) + (3.0 * x) + 1.0)
    f_jax = lambda x: jnp.sum((x * x) + (3.0 * x) + 1.0)

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_matmul_hessian(mode: str):
    cleanup_caches()

    x_jax = make_jax_array(3, seed=17)
    x_nb = tensor_from_jax(x_jax)

    w_jax = make_jax_array(2, 3, seed=19)
    w_nb = tensor_from_jax(w_jax)

    def f_nb(x):
        y = nb.matmul(w_nb, x)
        return nb.reduce_sum(y * y)

    def f_jax(x):
        y = w_jax @ x
        return jnp.sum(y * y)

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_reduce_sum_axis_hessian(mode: str):
    cleanup_caches()

    x_jax = make_jax_array(3, 4, seed=23)
    x_nb = tensor_from_jax(x_jax)

    f_nb = lambda x: nb.reduce_sum(nb.sin(nb.reduce_sum(x * x, axis=0)))
    f_jax = lambda x: jnp.sum(jnp.sin(jnp.sum(x * x, axis=0)))

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_mean_axis_hessian(mode: str):
    cleanup_caches()

    x_jax = make_jax_array(3, 4, seed=29)
    x_nb = tensor_from_jax(x_jax)

    f_nb = lambda x: nb.reduce_sum(nb.exp(nb.mean(x * x, axis=-1)))
    f_jax = lambda x: jnp.sum(jnp.exp(jnp.mean(x * x, axis=-1)))

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_reshape_hessian(mode: str):
    cleanup_caches()

    x_jax = make_jax_array(3, 4, seed=31)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.reshape(x, (12,))
        return nb.reduce_sum(y * y * y)

    def f_jax(x):
        y = jnp.reshape(x, (12,))
        return jnp.sum(y * y * y)

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_broadcast_to_hessian(mode: str):
    cleanup_caches()

    x_jax = make_jax_array(4, seed=37)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.broadcast_to(nb.sin(x), (2, 4))
        return nb.reduce_sum(y * y)

    def f_jax(x):
        y = jnp.broadcast_to(jnp.sin(x), (2, 4))
        return jnp.sum(y * y)

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_moveaxis_hessian(mode: str):
    cleanup_caches()

    x_jax = make_jax_array(3, 4, seed=41)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.moveaxis(x, source=0, destination=1)
        return nb.reduce_sum(y * y * y)

    def f_jax(x):
        y = jnp.moveaxis(x, 0, 1)
        return jnp.sum(y * y * y)

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_swap_axes_hessian(mode: str):
    cleanup_caches()

    x_jax = make_jax_array(3, 4, seed=43)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.swap_axes(x, axis1=0, axis2=1)
        return nb.reduce_sum(nb.sin(y) * y)

    def f_jax(x):
        y = jnp.swapaxes(x, 0, 1)
        return jnp.sum(jnp.sin(y) * y)

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_unsqueeze_squeeze_hessian(mode: str):
    cleanup_caches()

    x_jax = make_jax_array(4, seed=47)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.unsqueeze(x, axis=0)
        y = nb.squeeze(y, axis=0)
        return nb.reduce_sum(y * y * y)

    def f_jax(x):
        y = jnp.expand_dims(x, 0)
        y = jnp.squeeze(y, 0)
        return jnp.sum(y * y * y)

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)


@pytest.mark.parametrize("mode", ["fwd_grad", "rev_grad"])
def test_getitem_scalar_hessian(mode: str):
    """Direct check for indexing path used by tutorial scalar Hessian example."""
    cleanup_caches()

    x_jax = jnp.array([2.0, 3.0], dtype=jnp.float32)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        return x[0] ** 2 * x[1] + x[1] ** 3

    def f_jax(x):
        return x[0] ** 2 * x[1] + x[1] ** 3

    _assert_hessian_close(f_nb, f_jax, x_nb, x_jax, mode=mode)
