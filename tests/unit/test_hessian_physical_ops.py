# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Hessian tests focused on physical ops and implicit physical paths.

This suite is intentionally op-centric to isolate failures in VJP/JVP rules for
physical axis handling and batch-dim aware kernels.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb
from nabla.ops.reduction import reduce_max_physical, reduce_min_physical
from nabla.ops.view import broadcast_batch_dims, moveaxis_physical
from tests.unit.common import cleanup_caches, make_jax_array, tensor_from_jax, to_jax


MODES = ("rev_rev", "fwd_fwd", "rev_fwd", "fwd_rev")


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


def _assert_hessian_close_all_modes(
    f_nb: Callable,
    f_jax: Callable,
    x_nb,
    x_jax,
    *,
    rtol: float = 3e-3,
    atol: float = 3e-3,
):
    failures: list[str] = []
    for mode in MODES:
        try:
            h_nb = _hessian_nb(mode, f_nb)(x_nb)
            h_jax = _hessian_jax(mode, f_jax)(x_jax)
            np.testing.assert_allclose(
                to_jax(h_nb), np.array(h_jax), rtol=rtol, atol=atol
            )
        except Exception as exc:  # pragma: no cover - diagnostic failure path
            failures.append(f"{mode}: {type(exc).__name__}: {exc}")

    if failures:
        joined = "\n".join(failures)
        pytest.fail(f"Physical-op Hessian failures:\n{joined}")


def _assert_modes_consistent(
    f_nb: Callable,
    x_nb,
    *,
    rtol: float = 3e-3,
    atol: float = 3e-3,
):
    results: dict[str, np.ndarray] = {}
    failures: list[str] = []
    for mode in MODES:
        try:
            h_nb = _hessian_nb(mode, f_nb)(x_nb)
            results[mode] = np.array(to_jax(h_nb))
        except Exception as exc:  # pragma: no cover - diagnostic failure path
            failures.append(f"{mode}: {type(exc).__name__}: {exc}")

    if failures:
        joined = "\n".join(failures)
        pytest.fail(f"Physical-op mode execution failures:\n{joined}")

    base = results["rev_rev"]
    for mode in ("fwd_fwd", "rev_fwd", "fwd_rev"):
        np.testing.assert_allclose(results[mode], base, rtol=rtol, atol=atol)


@pytest.mark.parametrize("mode", MODES)
def test_hessian_moveaxis_physical(mode: str):
    cleanup_caches()
    x_jax = make_jax_array(2, 3, 4, seed=301)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = moveaxis_physical(x, source=0, destination=2)
        return nb.reduce_sum(y * y)

    def f_jax(x):
        y = jnp.moveaxis(x, 0, 2)
        return jnp.sum(y * y)

    h_nb = _hessian_nb(mode, f_nb)(x_nb)
    h_jax = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(h_nb), np.array(h_jax), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", MODES)
def test_hessian_unsqueeze_squeeze_physical(mode: str):
    cleanup_caches()
    x_jax = make_jax_array(3, 4, seed=303)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.unsqueeze_physical(x, axis=0)
        y = nb.squeeze_physical(y, axis=0)
        return nb.reduce_sum(y * y * y)

    def f_jax(x):
        y = jnp.expand_dims(x, axis=0)
        y = jnp.squeeze(y, axis=0)
        return jnp.sum(y * y * y)

    h_nb = _hessian_nb(mode, f_nb)(x_nb)
    h_jax = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(h_nb), np.array(h_jax), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", MODES)
def test_hessian_broadcast_to_physical(mode: str):
    cleanup_caches()
    x_jax = make_jax_array(2, 3, seed=307)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.broadcast_to_physical(x, (4, 2, 3))
        return nb.reduce_sum(y * y)

    def f_jax(x):
        y = jnp.broadcast_to(x, (4, 2, 3))
        return jnp.sum(y * y)

    h_nb = _hessian_nb(mode, f_nb)(x_nb)
    h_jax = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(h_nb), np.array(h_jax), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", MODES)
def test_hessian_reduce_sum_physical(mode: str):
    cleanup_caches()
    x_jax = make_jax_array(2, 3, 4, seed=311)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.reduce_sum_physical(x * x, axis=1, keepdims=False)
        return nb.reduce_sum(y * y)

    def f_jax(x):
        y = jnp.sum(x * x, axis=1)
        return jnp.sum(y * y)

    h_nb = _hessian_nb(mode, f_nb)(x_nb)
    h_jax = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(h_nb), np.array(h_jax), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("mode", MODES)
def test_hessian_mean_physical(mode: str):
    cleanup_caches()
    x_jax = make_jax_array(2, 3, 4, seed=313)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.mean_physical(x * x, axis=2, keepdims=False)
        return nb.reduce_sum(nb.exp(y))

    def f_jax(x):
        y = jnp.mean(x * x, axis=2)
        return jnp.sum(jnp.exp(y))

    h_nb = _hessian_nb(mode, f_nb)(x_nb)
    h_jax = _hessian_jax(mode, f_jax)(x_jax)
    np.testing.assert_allclose(to_jax(h_nb), np.array(h_jax), rtol=4e-3, atol=4e-3)


def test_hessian_implicit_broadcast_batch_dims_chain():
    """Implicit physical path: broadcast_batch_dims + reduce_sum_physical.

    This has no direct JAX equivalent with Nabla batch_dims metadata, so we
    assert consistency across all 4 Hessian constructions.
    """
    cleanup_caches()
    x_jax = make_jax_array(2, 4, seed=317)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = broadcast_batch_dims(x, (3,))
        y = nb.reduce_sum_physical(y * y, axis=0, keepdims=False)
        return nb.reduce_sum(y)

    _assert_modes_consistent(f_nb, x_nb, rtol=4e-3, atol=4e-3)


def test_hessian_implicit_move_axis_batch_chain():
    """Implicit physical path: move axis to/from batch dims helpers."""
    cleanup_caches()
    x_jax = make_jax_array(2, 3, 4, seed=319)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = nb.move_axis_to_batch_dims(x, axis=1)
        y = nb.move_axis_from_batch_dims(y, batch_axis=0, logical_destination=1)
        return nb.reduce_sum(y * y)

    def f_jax(x):
        # Equivalent logical permutation round-trip.
        y = jnp.moveaxis(x, 1, 0)
        y = jnp.moveaxis(y, 0, 1)
        return jnp.sum(y * y)

    _assert_hessian_close_all_modes(f_nb, f_jax, x_nb, x_jax, rtol=3e-3, atol=3e-3)


def test_hessian_reduce_max_physical_modes_smoke():
    """Non-smooth op: require all Hessian constructions to execute consistently."""
    cleanup_caches()
    x_jax = make_jax_array(2, 3, 4, seed=331)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = reduce_max_physical(x * x + 0.1, axis=2, keepdims=False)
        return nb.reduce_sum(y)

    _assert_modes_consistent(f_nb, x_nb, rtol=5e-3, atol=5e-3)


def test_hessian_reduce_min_physical_modes_smoke():
    """Non-smooth op: require all Hessian constructions to execute consistently."""
    cleanup_caches()
    x_jax = make_jax_array(2, 3, 4, seed=337)
    x_nb = tensor_from_jax(x_jax)

    def f_nb(x):
        y = reduce_min_physical(x * x + 0.1, axis=1, keepdims=False)
        return nb.reduce_sum(y)

    _assert_modes_consistent(f_nb, x_nb, rtol=5e-3, atol=5e-3)
