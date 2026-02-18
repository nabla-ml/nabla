# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Contract tests for Jacobian basis lifting under active batch prefixes."""

from __future__ import annotations

import jax.numpy as jnp

import nabla as nb
from nabla.ops.view.batch import broadcast_batch_dims
from nabla.transforms.utils import lift_basis_to_batch_prefix, std_basis
from tests.unit.common import cleanup_caches, tensor_from_jax


def _physical_prefix(t, n):
    phys = t.physical_global_shape or t.local_shape
    return tuple(int(d) for d in phys[:n])


def test_lift_basis_to_batch_prefix_matches_reference_prefix():
    cleanup_caches()

    ref = tensor_from_jax(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
    ref = broadcast_batch_dims(ref, (4,))

    _, basis = std_basis([ref])
    assert basis[0].batch_dims == 0

    lifted = lift_basis_to_batch_prefix(basis, [ref])

    assert lifted[0].batch_dims == ref.batch_dims
    assert _physical_prefix(lifted[0], ref.batch_dims) == _physical_prefix(
        ref, ref.batch_dims
    )


def test_lift_basis_does_not_overwrite_already_lifted_basis():
    cleanup_caches()

    ref = tensor_from_jax(jnp.array([1.0, 2.0], dtype=jnp.float32))
    ref = broadcast_batch_dims(ref, (3,))

    _, basis = std_basis([ref])
    lifted_once = lift_basis_to_batch_prefix(basis, [ref])
    lifted_twice = lift_basis_to_batch_prefix(lifted_once, [ref])

    assert lifted_twice[0].batch_dims == ref.batch_dims
    assert _physical_prefix(lifted_twice[0], ref.batch_dims) == _physical_prefix(
        ref, ref.batch_dims
    )


def test_vmapped_jacobian_smoke_respects_batch_prefix_for_basis():
    cleanup_caches()

    x = tensor_from_jax(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32))

    def f(v):
        return nb.reduce_sum(v * v * v)

    jac_rev = nb.vmap(nb.jacrev(f))(x)
    jac_fwd = nb.vmap(nb.jacfwd(f))(x)

    assert jac_rev.batch_dims == 0
    assert jac_fwd.batch_dims == 0
    assert tuple(int(d) for d in jac_rev.shape) == (2, 2)
    assert tuple(int(d) for d in jac_fwd.shape) == (2, 2)
