# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Vectorizing map (vmap) — automatic batching transform.

Usage matches JAX::

    batched_fn = vmap(fn, in_axes=0, out_axes=0)
    result = batched_fn(batched_x, batched_y)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, Union

if TYPE_CHECKING:
    from ..core.sharding.spec import DeviceMesh
    from ..core.tensor.api import Tensor
    from max.graph.dim import Dim

AxisSpec = Union[
    int, None, dict[str, "AxisSpec"], list["AxisSpec"], tuple["AxisSpec", ...]
]
T = TypeVar("T")


# ── Axis-spec helpers ──────────────────────────────────────────────────────


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis >= 0:
        if axis >= ndim:
            raise ValueError(f"Axis {axis} out of bounds for {ndim}-d tensor")
        return axis
    n = ndim + axis
    if n < 0:
        raise ValueError(f"Axis {axis} out of bounds for {ndim}-d tensor")
    return n


def _broadcast_to_args(spec: AxisSpec, n: int) -> tuple[AxisSpec, ...]:
    """Broadcast a single axis spec to *n* arguments."""
    if isinstance(spec, (int, type(None))):
        return (spec,) * n
    if isinstance(spec, (list, tuple)):
        if len(spec) != n:
            raise ValueError(
                f"in_axes/out_axes length ({len(spec)}) != number of args ({n})"
            )
        return tuple(spec)
    if isinstance(spec, dict):
        if n == 1:
            return (spec,)
        raise TypeError(
            f"Dict axis spec with {n} args — use tuple/list for multiple args"
        )
    raise TypeError(f"Invalid axis spec type: {type(spec).__name__}")


# ── Prefix-pytree traversal ───────────────────────────────────────────────
# JAX's "prefix pytree": a scalar spec (int/None) broadcasts to every leaf;
# a container spec must structurally match the data tree.


def _walk_prefix(
    fn: Callable[..., T],
    tree: Any,
    prefix: AxisSpec,
    *extra: Any,
) -> Any:
    """Apply *fn(tensor, axis, *extra)* to every Tensor leaf paired with its
    axis from *prefix*.  Returns the mapped tree (same structure as *tree*)."""
    from ..core.tensor import Tensor
    from ..core import tree_map

    # Leaf spec → broadcast to all tensor leaves.
    if isinstance(prefix, (int, type(None))):
        return tree_map(
            lambda leaf: fn(leaf, prefix, *extra) if isinstance(leaf, Tensor) else leaf,
            tree,
        )

    # Direct tensor leaf.
    if isinstance(tree, Tensor):
        return fn(tree, prefix, *extra)

    # Structured prefix → walk both in lockstep via pytree semantics.
    return tree_map(
        lambda leaf, axis: fn(leaf, axis, *extra) if isinstance(leaf, Tensor) else leaf,
        tree,
        prefix,
    )


def _collect_prefix(fn: Callable, tree: Any, prefix: AxisSpec) -> list:
    """Collect non-None results of *fn(tensor, axis)* across all leaves."""
    from ..core.tensor import Tensor
    from ..core import tree_map

    out: list = []

    if isinstance(prefix, (int, type(None))):
        tree_map(
            lambda leaf: out.append(v)
            if isinstance(leaf, Tensor) and (v := fn(leaf, prefix)) is not None
            else None,
            tree,
        )
        return out

    tree_map(
        lambda leaf, axis: out.append(v)
        if isinstance(leaf, Tensor) and (v := fn(leaf, axis)) is not None
        else None,
        tree,
        prefix,
    )
    return out


# ── Batch-size validation ─────────────────────────────────────────────────


def _get_batch_size(tensor: "Tensor", axis: AxisSpec) -> "Dim":
    if axis is None:
        return None
    if not isinstance(axis, int):
        raise TypeError(
            f"Expected int|None for axis at leaf, got {type(axis).__name__}"
        )
    shape = tensor.shape
    if shape.rank == 0:
        raise ValueError(
            f"Cannot batch scalar (rank=0) along axis {axis}. Use in_axes=None."
        )
    return shape[_normalize_axis(axis, shape.rank)]


def _validate_batch_sizes(
    args: tuple[Any, ...], in_axes: tuple[AxisSpec, ...], axis_size: int | None
) -> "Dim":
    """Validate batch dims across all inputs; return common Dim."""
    from max.graph.dim import StaticDim

    dims = [
        d
        for d in (
            d
            for arg, ax in zip(args, in_axes)
            for d in _collect_prefix(_get_batch_size, arg, ax)
        )
        if d is not None
    ]

    if not dims:
        if axis_size is not None:
            return StaticDim(axis_size)
        raise ValueError("All in_axes are None. Must specify axis_size.")

    first = dims[0]
    if not all(d == first for d in dims):
        raise ValueError(f"Inconsistent batch dims: {dims}")
    if axis_size is not None:
        expected = StaticDim(axis_size)
        if first != expected:
            raise ValueError(f"axis_size={axis_size} doesn't match inferred {first}")
    return first


# ── Batch / unbatch ───────────────────────────────────────────────────────


def _batch_tensor(
    tensor: "Tensor",
    axis: AxisSpec,
    batch_dim: "Dim",
    spmd_axis_name: str | None,
    mesh: "DeviceMesh | None",
) -> "Tensor":
    """Move *axis* into batch_dims position for batched execution."""
    from max.graph.dim import StaticDim
    from ..ops import communication as comm_ops
    from ..ops import view as view_ops

    old_bd = tensor.batch_dims

    if axis is None:
        t = view_ops.unsqueeze(tensor, axis=0)
        target_shape = (batch_dim,) + tuple(tensor.shape)
        if not (isinstance(batch_dim, StaticDim) and batch_dim.dim == 1):
            t = view_ops.broadcast_to(t, shape=target_shape)
        t = view_ops.incr_batch_dims(t)
        if old_bd > 0:
            t = view_ops.moveaxis_physical(t, source=old_bd, destination=0)
        if spmd_axis_name is not None and mesh is not None:
            t = _apply_shard(t, tensor, old_bd, spmd_axis_name, mesh, broadcast=True)
    else:
        if axis != 0:
            lr = len(tensor.shape)
            na = axis if axis >= 0 else lr + axis
            t = view_ops.moveaxis_physical(
                tensor, source=old_bd + na, destination=old_bd
            )
        else:
            t = tensor
        if spmd_axis_name is not None and mesh is not None:
            t = _apply_shard(t, t, old_bd, spmd_axis_name, mesh, broadcast=False)
        t = view_ops.incr_batch_dims(t)
        if old_bd > 0:
            t = view_ops.moveaxis_physical(t, source=old_bd, destination=0)

    return t


def _apply_shard(
    target: "Tensor",
    source: "Tensor",
    bd_offset: int,
    axis_name: str,
    mesh: "DeviceMesh",
    *,
    broadcast: bool,
) -> "Tensor":
    """Apply SPMD sharding spec to a batched dim."""
    from ..core.sharding.spec import DimSpec
    from ..ops import communication as comm_ops

    if broadcast:
        pr = bd_offset + 1 + len(target.shape)
        specs = [DimSpec([]) for _ in range(pr)]
        if source.sharding:
            for i in range(len(source.sharding.dim_specs)):
                if i + 1 < len(specs):
                    specs[i + 1] = source.sharding.dim_specs[i].clone()
        specs[0] = DimSpec([axis_name])
    else:
        lr = len(target.shape)
        pr = bd_offset + lr
        specs = [DimSpec([]) for _ in range(pr)]
        if target.sharding:
            for i in range(min(pr, len(target.sharding.dim_specs))):
                specs[i] = target.sharding.dim_specs[i].clone()
        specs[bd_offset] = DimSpec([axis_name])

    return comm_ops.shard(target, mesh=mesh, dim_specs=specs)


def _unbatch_tensor(
    tensor: "Tensor",
    axis: AxisSpec,
    spmd_axis_name: str | None = None,
    mesh: "DeviceMesh | None" = None,
) -> "Tensor":
    """Restore *tensor* after batched execution — move batch_dims to *axis*."""
    from ..ops import view as view_ops

    cbd = tensor.batch_dims
    t = (
        view_ops.moveaxis_physical(tensor, source=0, destination=cbd - 1)
        if cbd > 1
        else tensor
    )
    t = view_ops.decr_batch_dims(t)

    if axis is None:
        return view_ops.squeeze(t, axis=0)
    if axis != 0:
        nbd = t.batch_dims
        lr = len(t.shape)
        na = axis if axis >= 0 else lr + axis
        t = view_ops.moveaxis_physical(t, source=nbd, destination=nbd + na)
    return t


# ── Public API ─────────────────────────────────────────────────────────────


def vmap(
    func: Callable[..., T] | None = None,
    in_axes: AxisSpec = 0,
    out_axes: AxisSpec = 0,
    axis_size: int | None = None,
    spmd_axis_name: str | None = None,
    mesh: "DeviceMesh | None" = None,
) -> Callable[..., T]:
    """Vectorize *func* over batch dimensions (JAX-compatible API)."""
    if func is None:
        return lambda f: vmap(
            f,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            mesh=mesh,
        )

    def vectorized(*args: Any) -> Any:
        if not args:
            raise ValueError("vmap requires at least one input argument.")

        in_ax = _broadcast_to_args(in_axes, len(args))
        batch_dim = _validate_batch_sizes(args, in_ax, axis_size)

        batched = tuple(
            _walk_prefix(_batch_tensor, arg, ax, batch_dim, spmd_axis_name, mesh)
            for arg, ax in zip(args, in_ax)
        )

        outputs = func(*batched)

        is_single = not isinstance(outputs, (list, tuple))
        out_list = [outputs] if is_single else list(outputs)
        out_ax = _broadcast_to_args(out_axes, len(out_list))

        unbatched = [
            _walk_prefix(_unbatch_tensor, out, ax, spmd_axis_name, mesh)
            for out, ax in zip(out_list, out_ax)
        ]
        return unbatched[0] if is_single else type(outputs)(unbatched)

    return vectorized
