# ===----------------------------------------------------------------------=== #
# Nabla 2026 — Unified Op × Transform Test Registry
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp

import nabla as nb

from .common import (
    OpConfig,
    Operation,
    get_shape_for_rank,
    get_test_data_for_shapes,
    standard_get_args,
    tensor_from_jax,
)

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


def _jax_split(x, *, num_splits: int, axis: int = 0):
    return tuple(jnp.split(x, num_splits, axis=axis))


def _jax_chunk(x, *, chunks: int, axis: int = 0):
    return list(jnp.split(x, chunks, axis=axis))


def _jax_unbind(x, *, axis: int = 0):
    splits = jnp.split(x, x.shape[axis], axis=axis)
    return tuple(jnp.squeeze(s, axis=axis) for s in splits)


def _jax_slice_tensor(x, *, start, size):
    slices = tuple(slice(s, s + sz) for s, sz in zip(start, size, strict=False))
    return x[slices]


def _jax_slice_update(x, update, *, start, size):
    slices = tuple(slice(s, s + sz) for s, sz in zip(start, size, strict=False))
    return x.at[slices].set(update)


def _jax_gather(x, indices, *, axis: int = 0):
    return jnp.take(x, indices, axis=axis)


def _jax_scatter(x, indices, updates, *, axis: int = 0):
    if axis != 0:
        raise ValueError("_jax_scatter only supports axis=0 in tests")
    return x.at[indices].set(updates)


def _jax_permute(x, *, order):
    return jnp.transpose(x, axes=order)


def _jax_pad(x, *, paddings, mode="constant", value=0.0):
    return jnp.pad(x, paddings, mode=mode, constant_values=value)


def _list_input_get_args(config: OpConfig):
    shapes = config.primal_shapes or tuple(get_shape_for_rank(r) for r in config.ranks)
    primals_nb, primals_jax = get_test_data_for_shapes(shapes, config)
    return ((list(primals_nb),), config.params), ((list(primals_jax),), config.params)


def _where_get_args(config: OpConfig):
    shape = config.primal_shapes[0]
    primals_nb, primals_jax = get_test_data_for_shapes((shape, shape), config)
    condition = (jnp.arange(math.prod(shape)).reshape(shape) % 2 == 0)
    condition_nb = tensor_from_jax(condition)
    return (
        ((condition_nb, primals_nb[0], primals_nb[1]), config.params),
        ((condition, primals_jax[0], primals_jax[1]), config.params),
    )


def _gather_get_args(config: OpConfig):
    x_shape, indices_shape = config.primal_shapes
    primals_nb, primals_jax = get_test_data_for_shapes((x_shape,), config)
    axis = config.params.get("axis", 0)
    axis_size = x_shape[axis]
    indices = (jnp.arange(math.prod(indices_shape)) % axis_size).reshape(indices_shape)
    indices = indices.astype(jnp.int64)
    indices_nb = tensor_from_jax(indices)
    return (
        ((primals_nb[0], indices_nb), config.params),
        ((primals_jax[0], indices), config.params),
    )


def _scatter_get_args(config: OpConfig):
    x_shape, indices_shape, updates_shape = config.primal_shapes
    primals_nb, primals_jax = get_test_data_for_shapes((x_shape, updates_shape), config)
    axis = config.params.get("axis", 0)
    if axis != 0:
        raise ValueError("scatter test args only support axis=0")
    axis_size = x_shape[axis]
    indices = (jnp.arange(math.prod(indices_shape)) % axis_size).reshape(indices_shape)
    indices = indices.astype(jnp.int64)
    indices_nb = tensor_from_jax(indices)
    return (
        ((primals_nb[0], indices_nb, primals_nb[1]), config.params),
        ((primals_jax[0], indices, primals_jax[1]), config.params),
    )


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
    Operation("concatenate", "view",
              nb.concatenate, jnp.concatenate,
              [OpConfig(description="axis1", primal_shapes=((2, 3), (2, 1)), params={"axis": 1}, is_list_input=True)],
              _list_input_get_args),
    Operation("stack", "view",
              nb.stack, jnp.stack,
              [OpConfig(description="axis0", primal_shapes=((2, 3), (2, 3)), params={"axis": 0}, is_list_input=True)],
              _list_input_get_args),
    Operation("slice_tensor", "view",
              nb.slice_tensor, _jax_slice_tensor,
              [OpConfig(description="2d", primal_shapes=((4, 5),), params={"start": [1, 2], "size": [2, 2]})],
              standard_get_args),
    Operation("slice_update", "view",
              nb.slice_update, _jax_slice_update,
              [OpConfig(description="2d", primal_shapes=((4, 5), (2, 2)), params={"start": [1, 2], "size": [2, 2]})],
              standard_get_args),
    Operation("permute", "view",
              partial(nb.permute, order=(2, 0, 1)), partial(_jax_permute, order=(2, 0, 1)),
              [OpConfig(description="3d", ranks=(3,))],
              standard_get_args),
    Operation("pad", "view",
              partial(nb.pad, paddings=[(1, 1), (2, 0)], mode="constant", value=0.0),
              partial(_jax_pad, paddings=[(1, 1), (2, 0)], mode="constant", value=0.0),
              [OpConfig(description="2d", ranks=(2,))],
              standard_get_args),
    Operation("broadcast_to", "view",
              partial(nb.broadcast_to, shape=(2, 1, 3)),
              partial(jnp.broadcast_to, shape=(2, 1, 3)),
              [OpConfig(description="rank2_to3", primal_shapes=((1, 3),))],
              standard_get_args),
]

# ---------------------------------------------------------------------------
# ALL OPS
# ---------------------------------------------------------------------------
MULTI_OUTPUT_OPS: list[Operation] = [
    Operation("split", "multi_output",
              nb.split, _jax_split,
              [OpConfig(description="axis1_3", primal_shapes=((4, 6),), params={"num_splits": 3, "axis": 1})],
              standard_get_args),
    Operation("chunk", "multi_output",
              nb.chunk, _jax_chunk,
              [OpConfig(description="axis0_3", primal_shapes=((6, 4),), params={"chunks": 3, "axis": 0})],
              standard_get_args),
    Operation("unbind", "multi_output",
              nb.unbind, _jax_unbind,
              [OpConfig(description="axis0", primal_shapes=((3, 4, 2),), params={"axis": 0})],
              standard_get_args),
]

CONTROL_FLOW_OPS: list[Operation] = [
    Operation("where", "control_flow",
              nb.where, jnp.where,
              [OpConfig(description="rank2", primal_shapes=((4, 4),), params={})],
              _where_get_args),
]

INDEXING_OPS: list[Operation] = [
    Operation("gather", "indexing",
              nb.gather, _jax_gather,
              [OpConfig(description="axis1", primal_shapes=((4, 5), (2, 2)), params={"axis": 1},
                        supports_sharding=False)],
              _gather_get_args),
    Operation("scatter", "indexing",
              nb.scatter, _jax_scatter,
              [OpConfig(description="axis0", primal_shapes=((4, 5), (2,), (2, 5)), params={"axis": 0},
                        supports_sharding=False)],
              _scatter_get_args),
]

OUTER_OPS: list[Operation] = [
    Operation("outer", "binary",
              nb.outer, jnp.outer,
              [OpConfig(description="vectors", primal_shapes=((3,), (3,)))],
              standard_get_args),
]

ALL_OPS = (
    UNARY_OPS
    + BINARY_OPS
    + MATMUL_OPS
    + REDUCTION_OPS
    + VIEW_OPS
    + MULTI_OUTPUT_OPS
    + CONTROL_FLOW_OPS
    + INDEXING_OPS
    + OUTER_OPS
)
