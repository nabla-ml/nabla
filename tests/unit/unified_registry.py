# ===----------------------------------------------------------------------=== #
# Nabla 2026 — Unified Op × Transform Test Registry
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

import nabla as nb

from .common import OpConfig, Operation, standard_get_args

# ---------------------------------------------------------------------------
# JAX compatibility wrappers
# ---------------------------------------------------------------------------

def _jax_matmul(x, y):
    if x.ndim == 1 and y.ndim == 1:
        return jnp.dot(x, y)
    if x.ndim == 1:
        return (jnp.expand_dims(x, 0) @ y).squeeze(0)
    if y.ndim == 1:
        return (x @ jnp.expand_dims(y, 1)).squeeze(-1)
    return jnp.matmul(x, y)


def _jax_swap_axes(x, axis1, axis2):
    rank = len(x.shape)
    axes = list(range(rank))
    a1 = axis1 if axis1 >= 0 else rank + axis1
    a2 = axis2 if axis2 >= 0 else rank + axis2
    axes[a1], axes[a2] = axes[a2], axes[a1]
    return jnp.transpose(x, axes=axes)


def _jax_unsqueeze(x, axes):
    for a in sorted(axes if isinstance(axes, (list, tuple)) else [axes]):
        x = jnp.expand_dims(x, axis=a)
    return x


# ---------------------------------------------------------------------------
# UNARY OPS (minimal)
# ---------------------------------------------------------------------------

_UNARY_RANKS = (1, 2, 3)


def _unary_configs():
    return [OpConfig(description=f"rank{r}", ranks=(r,)) for r in _UNARY_RANKS]


UNARY_OPS: list[Operation] = [
    Operation("sin", "unary", nb.sin, jnp.sin, _unary_configs(), standard_get_args),
    Operation("tanh", "unary", nb.tanh, jnp.tanh, _unary_configs(), standard_get_args),
]

# ---------------------------------------------------------------------------
# BINARY OPS (minimal)
# ---------------------------------------------------------------------------

_BINARY_RANKS = ((2, 2), (3, 3))


def _binary_configs():
    configs = [
        OpConfig(description=f"r{r1}xr{r2}", ranks=(r1, r2))
        for r1, r2 in _BINARY_RANKS
    ]
    configs.append(
        OpConfig(description="broadcast", primal_shapes=((4, 4), (1, 4)))
    )
    return configs


BINARY_OPS: list[Operation] = [
    Operation("add", "binary", nb.add, jnp.add, _binary_configs(), standard_get_args),
    Operation("mul", "binary", nb.mul, jnp.multiply, _binary_configs(), standard_get_args),
]

# ---------------------------------------------------------------------------
# MATMUL
# ---------------------------------------------------------------------------

MATMUL_OPS: list[Operation] = [
    Operation("matmul", "binary",
              nb.matmul, _jax_matmul,
              [
                  OpConfig(description="2d", primal_shapes=((4, 8), (8, 4))),
                  OpConfig(description="batch", primal_shapes=((2, 4, 8), (2, 8, 4))),
              ],
              standard_get_args),
]

# ---------------------------------------------------------------------------
# REDUCTION OPS
# ---------------------------------------------------------------------------

REDUCTION_OPS: list[Operation] = [
    Operation("reduce_sum", "reduction",
              partial(nb.reduce_sum, axis=-1), partial(jnp.sum, axis=-1),
              [OpConfig(description=f"rank{r}", ranks=(r,)) for r in (2, 3)],
              standard_get_args),
    Operation("mean", "reduction",
              partial(nb.mean, axis=-1), partial(jnp.mean, axis=-1),
              [OpConfig(description=f"rank{r}", ranks=(r,)) for r in (2, 3)],
              standard_get_args),
    Operation("reduce_max", "reduction",
              partial(nb.reduce_max, axis=-1), partial(jnp.max, axis=-1),
              [OpConfig(description=f"rank{r}", ranks=(r,), use_stable_floats=True) for r in (2, 3)],
              standard_get_args),
]

# ---------------------------------------------------------------------------
# VIEW OPS
# ---------------------------------------------------------------------------

VIEW_OPS: list[Operation] = [
    Operation("reshape", "view",
              lambda x: nb.reshape(x, (int(x.shape[0]) * int(x.shape[1]),)),
              lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],)),
              [OpConfig(description="2d", ranks=(2,))],
              standard_get_args),
    Operation("swap_axes", "view",
              partial(nb.swap_axes, axis1=0, axis2=1),
              partial(_jax_swap_axes, axis1=0, axis2=1),
              [OpConfig(description="2d", ranks=(2,)),
               OpConfig(description="3d", ranks=(3,))],
              standard_get_args),
    Operation("unsqueeze", "view",
              partial(nb.unsqueeze, axis=0),
              partial(_jax_unsqueeze, axes=0),
              [OpConfig(description="2d", ranks=(2,))],
              standard_get_args),
    Operation("squeeze", "view",
              partial(nb.squeeze, axis=0),
              partial(jnp.squeeze, axis=0),
              [OpConfig(description="squeezable", primal_shapes=((1, 4, 4),))],
              standard_get_args),
    Operation("moveaxis", "view",
              partial(nb.moveaxis, source=0, destination=-1),
              partial(jnp.moveaxis, source=0, destination=-1),
              [OpConfig(description="3d", ranks=(3,))],
              standard_get_args),
]

# ---------------------------------------------------------------------------
# ALL OPS
# ---------------------------------------------------------------------------

ALL_OPS = UNARY_OPS + BINARY_OPS + MATMUL_OPS + REDUCTION_OPS + VIEW_OPS
