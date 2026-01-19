# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tensor import Tensor
    from ..core.sharding.mesh import DeviceMesh



AxisSpec = Union[int, None, dict[str, "AxisSpec"], list["AxisSpec"], tuple["AxisSpec", ...]]

T = TypeVar("T")




def _is_leaf(obj: Any) -> bool:
    """Check if obj is a leaf (non-container) in axis specification context."""
    return not isinstance(obj, (dict, list, tuple))


def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize negative axis and validate bounds."""
    if axis >= 0:
        if axis >= ndim:
            raise ValueError(f"Axis {axis} out of bounds for {ndim}-dimensional tensor")
        return axis
    normalized = ndim + axis
    if normalized < 0:
        raise ValueError(f"Axis {axis} out of bounds for {ndim}-dimensional tensor")
    return normalized


def _broadcast_to_args(spec: AxisSpec, n: int) -> tuple[AxisSpec, ...]:
    """Broadcast axis spec to match n function arguments."""
    if isinstance(spec, (int, type(None))):
        return (spec,) * n
    if isinstance(spec, (list, tuple)):
        if len(spec) != n:
            raise ValueError(
                f"in_axes/out_axes length ({len(spec)}) doesn't match "
                f"number of arguments ({n})"
            )
        return tuple(spec)
    if isinstance(spec, dict):
        # Dict at top level applies to a single dict argument
        if n == 1:
            return (spec,)
        raise TypeError(
            f"Dict axis spec with {n} args - use tuple/list for multiple args, "
            f"or pass a single dict argument"
        )
    raise TypeError(f"Invalid axis specification type: {type(spec).__name__}")




def _map_prefix(
    fn: Callable[..., T],
    tree: Any,
    prefix: AxisSpec,
    *extra_args: Any,
) -> Any:
    """Map fn over tree's Tensor leaves with corresponding axis from prefix.
    
    Implements JAX's prefix pytree semantics:
    - If prefix is a leaf (int/None), it broadcasts to all tensor leaves
    - If prefix is a container, it must structurally match tree
    """
    from ..core.tensor import Tensor
    
    if isinstance(tree, Tensor):
        return fn(tree, prefix, *extra_args)
    
    # If prefix is a leaf, broadcast to all tensor leaves in tree
    if _is_leaf(prefix):
        if isinstance(tree, dict):
            return {k: _map_prefix(fn, v, prefix, *extra_args) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)([_map_prefix(fn, t, prefix, *extra_args) for t in tree])
        return tree
    
    # Both are containers - recurse with matching structure
    if isinstance(tree, dict) and isinstance(prefix, dict):
        missing = set(tree.keys()) - set(prefix.keys())
        if missing:
            raise ValueError(f"Axis spec missing keys: {missing}")
        return {k: _map_prefix(fn, v, prefix[k], *extra_args) for k, v in tree.items()}
    
    if isinstance(tree, (list, tuple)) and isinstance(prefix, (list, tuple)):
        if len(tree) != len(prefix):
            raise ValueError(
                f"Axis spec length ({len(prefix)}) doesn't match "
                f"tree length ({len(tree)})"
            )
        return type(tree)([_map_prefix(fn, t, a, *extra_args) for t, a in zip(tree, prefix)])
    
    return tree


def _collect_from_prefix(
    fn: Callable[[Any, AxisSpec], T | None],
    tree: Any,
    prefix: AxisSpec,
) -> list[T]:
    """Collect non-None results from fn applied to all (tensor, axis) pairs."""
    from ..core.tensor import Tensor
    results: list[T] = []
    
    def _collect(tree_part: Any, prefix_part: AxisSpec) -> None:
        if isinstance(tree_part, Tensor):
            result = fn(tree_part, prefix_part)
            if result is not None:
                results.append(result)
        elif _is_leaf(prefix_part):
            if isinstance(tree_part, dict):
                for v in tree_part.values():
                    _collect(v, prefix_part)
            elif isinstance(tree_part, (list, tuple)):
                for t in tree_part:
                    _collect(t, prefix_part)
        elif isinstance(tree_part, dict) and isinstance(prefix_part, dict):
            for k, v in tree_part.items():
                if k in prefix_part:
                    _collect(v, prefix_part[k])
        elif isinstance(tree_part, (list, tuple)) and isinstance(prefix_part, (list, tuple)):
            for t, a in zip(tree_part, prefix_part):
                _collect(t, a)
    
    _collect(tree, prefix)
    return results




def _get_batch_size(tensor: Tensor, axis: AxisSpec):
    """Get batch dimension for a single tensor/axis pair.
    
    Returns the Dim object (StaticDim or SymbolicDim) at the specified axis.
    Returns None if axis is None.
    """
    if axis is None:
        return None
    if not isinstance(axis, int):
        raise TypeError(f"Expected int or None for axis at tensor leaf, got {type(axis).__name__}")
    
    shape = tensor.shape
    if shape.rank == 0:
        raise ValueError(
            f"Cannot batch scalar tensor (rank=0) along axis {axis}. "
            "Use in_axes=None to broadcast scalars."
        )
    normalized = _normalize_axis(axis, shape.rank)
    return shape[normalized]


def _validate_batch_sizes(args: tuple, in_axes: tuple[AxisSpec, ...], axis_size: int | None):
    """Validate batch dimensions and return the common batch Dim.
    
    Returns:
        Dim object (StaticDim or SymbolicDim) representing the batch dimension.
    """
    from max.graph.dim import StaticDim
    
    dims = []
    for arg, ax in zip(args, in_axes):
        dims.extend(_collect_from_prefix(_get_batch_size, arg, ax))
    
    dims = [d for d in dims if d is not None]
    
    if not dims:
        if axis_size is not None:
            return StaticDim(axis_size)
        raise ValueError(
            "All in_axes are None (broadcast). Must specify axis_size "
            "to determine batch dimension size."
        )
    
    first = dims[0]
    if not all(d == first for d in dims):
        raise ValueError(f"Inconsistent batch dimensions along specified axes: {dims}")
    
    if axis_size is not None:
        expected = StaticDim(axis_size)
        if first != expected:
            raise ValueError(f"axis_size={axis_size} doesn't match inferred batch dim {first}")
    
    return first




def _batch_tensor(tensor: Tensor, axis: AxisSpec, batch_dim, spmd_axis_name: str | None, mesh: "DeviceMesh | None") -> Tensor:
    """Prepare tensor for batched execution by moving axis to batch_dims."""
    from ..ops import view as l_ops
    from ..ops import view as p_ops
    from ..ops import communication as comm_ops
    from max.graph.dim import StaticDim
    
    old_batch_dims = tensor.batch_dims
    
    if axis is None:
        t = l_ops.unsqueeze(tensor, axis=0)
        target_shape = (batch_dim,) + tuple(tensor.shape)
        
        needs_broadcast = not (isinstance(batch_dim, StaticDim) and batch_dim.dim == 1)
        if needs_broadcast:
            t = l_ops.broadcast_to(t, shape=target_shape)
        
        t = p_ops.incr_batch_dims(t)
        
        if old_batch_dims > 0:
            t = p_ops.moveaxis(t, source=old_batch_dims, destination=0)
            
        if spmd_axis_name is not None and mesh is not None:
             from ..core.sharding.spec import DimSpec
             
             physical_rank = old_batch_dims + 1 + len(t.shape)
             dim_specs = [DimSpec([]) for _ in range(physical_rank)]
             
             if tensor.sharding:
                 for i in range(len(tensor.sharding.dim_specs)):
                     if i + 1 < len(dim_specs):
                        dim_specs[i + 1] = tensor.sharding.dim_specs[i].clone()
             
             dim_specs[0] = DimSpec([spmd_axis_name])
             
             t = comm_ops.shard_op(t, mesh, dim_specs)
    else:
        if axis != 0:
            logical_rank = len(tensor.shape)
            norm_axis = axis if axis >= 0 else logical_rank + axis
            physical_axis = old_batch_dims + norm_axis
            
            t = p_ops.moveaxis(tensor, source=physical_axis, destination=old_batch_dims)
        else:
            t = tensor
        
        if spmd_axis_name is not None and mesh is not None:
            from ..core.sharding.spec import DimSpec
            
            logical_rank = len(t.shape)
            physical_rank = old_batch_dims + logical_rank
            
            dim_specs = [DimSpec([]) for _ in range(physical_rank)]
            
            if t.sharding:
                for i in range(min(old_batch_dims, len(t.sharding.dim_specs))):
                    dim_specs[i] = t.sharding.dim_specs[i].clone()
            
            dim_specs[old_batch_dims] = DimSpec([spmd_axis_name])
            
            t = comm_ops.shard_op(t, mesh, dim_specs)
        
        t = p_ops.incr_batch_dims(t)
        
        if old_batch_dims > 0:
            t = p_ops.moveaxis(t, source=old_batch_dims, destination=0)

    return t


def _unbatch_tensor(tensor: Tensor, axis: AxisSpec, spmd_axis_name: str | None = None, mesh: "DeviceMesh | None" = None) -> Tensor:
    """Restore tensor after batched execution by moving batch_dims to axis.
    
    Note: spmd_axis_name sharding is PRESERVED on the output axis - we don't
    all_gather automatically.
    """
    from ..ops import view as l_ops
    from ..ops import view as p_ops
    
    current_batch_dims = tensor.batch_dims
    
    if current_batch_dims > 1:
        t = p_ops.moveaxis(tensor, source=0, destination=current_batch_dims - 1)
    else:
        t = tensor
    
    t = p_ops.decr_batch_dims(t)
    
    if axis is None:
        return l_ops.squeeze(t, axis=0)
    
    if axis != 0:
        new_batch_dims = t.batch_dims
        logical_rank = len(t.shape)
        
        norm_axis = axis if axis >= 0 else logical_rank + axis
        
        source = new_batch_dims
        destination = new_batch_dims + norm_axis
        t = p_ops.moveaxis(t, source=source, destination=destination)
    
    return t




def vmap(
    func: Callable[..., T] | None = None,
    in_axes: AxisSpec = 0,
    out_axes: AxisSpec = 0,
    axis_size: int | None = None,
    spmd_axis_name: str | None = None,
    mesh: "DeviceMesh | None" = None,
) -> Callable[..., T]:
    """Vectorize a function over batch dimensions.
    
    Creates a function that maps `func` over axes of its inputs, similar to
    JAX's vmap. Uses the nabla module's batch_dims mechanism for transparent
    vectorization without explicit loops.
    """
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes, out_axes=out_axes, axis_size=axis_size, spmd_axis_name=spmd_axis_name, mesh=mesh)
    
    def vectorized(*args: Any) -> Any:
        if not args:
            raise ValueError("vmap requires at least one input argument.")
        
        in_ax = _broadcast_to_args(in_axes, len(args))
        
        batch_dim = _validate_batch_sizes(args, in_ax, axis_size)
        
        batched = tuple(
            _map_prefix(_batch_tensor, arg, ax, batch_dim, spmd_axis_name, mesh)
            for arg, ax in zip(args, in_ax)
        )
        
        outputs = func(*batched)
        
        is_single = not isinstance(outputs, (list, tuple))
        out_list = [outputs] if is_single else list(outputs)
        out_ax = _broadcast_to_args(out_axes, len(out_list))
        
        unbatched = [
            _map_prefix(_unbatch_tensor, out, ax, spmd_axis_name, mesh)
            for out, ax in zip(out_list, out_ax)
        ]
        
        return unbatched[0] if is_single else type(outputs)(unbatched)
    
    return vectorized


__all__ = ["vmap", "AxisSpec"]
