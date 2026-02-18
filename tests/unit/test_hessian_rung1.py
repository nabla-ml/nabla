from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from tests.unit.common import cleanup_caches, make_jax_array, tensor_from_jax, to_jax

MODES = ("rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev")

def _hessian_nb(mode: str, f):
    if mode == "rev_rev":
        return nb.jacrev(nb.jacrev(f))
    if mode == "fwd_fwd":
        return nb.jacfwd(nb.jacfwd(f))
    if mode == "rev_fwd":
        return nb.jacrev(nb.jacfwd(f))
    if mode == "fwd_rev":
        return nb.jacfwd(nb.jacrev(f))
    raise ValueError(mode)

def _hessian_jax(mode: str, f):
    if mode == "rev_rev":
        return jax.jacrev(jax.jacrev(f))
    if mode == "fwd_fwd":
        return jax.jacfwd(jax.jacfwd(f))
    if mode == "rev_fwd":
        return jax.jacrev(jax.jacfwd(f))
    if mode == "fwd_rev":
        return jax.jacfwd(jax.jacrev(f))
    raise ValueError(mode)

@pytest.mark.parametrize("mode", MODES)
def test_hessian_sum_exp(mode: str):
    cleanup_caches()
    x_jax = make_jax_array(5, seed=42)
    x_nb = tensor_from_jax(x_jax)

    f_nb = lambda x: nb.reduce_sum(nb.exp(x))
    f_jax = lambda x: jnp.sum(jnp.exp(x))

    h_nb = _hessian_nb(mode, f_nb)(x_nb)
    h_jax = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(h_nb), np.array(h_jax), rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("mode", MODES)
def test_hessian_cubic_poly(mode: str):
    cleanup_caches()
    x_jax = make_jax_array(4, seed=43)
    x_nb = tensor_from_jax(x_jax)

    # f(x) = sum(x^3 + 2x^2 - 5x)
    # f'(x) = 3x^2 + 4x - 5
    # f''(x) = 6x + 4 (diagonal)
    f_nb = lambda x: nb.reduce_sum(x * x * x + 2.0 * x * x - 5.0 * x)
    f_jax = lambda x: jnp.sum(x * x * x + 2.0 * x * x - 5.0 * x)

    h_nb = _hessian_nb(mode, f_nb)(x_nb)
    h_jax = _hessian_jax(mode, f_jax)(x_jax)
    print(f"\nDEBUG: {mode} x={x_jax}")
    print(f"DEBUG: {mode} NB Hessian diagonal={np.diag(to_jax(h_nb))}")
    print(f"DEBUG: {mode} JAX Hessian diagonal={np.diag(np.array(h_jax))}")
    np.testing.assert_allclose(to_jax(h_nb), np.array(h_jax), rtol=1e-3, atol=1e-3)
