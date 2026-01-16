# ===----------------------------------------------------------------------=== #
# Nabla 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Function transforms for the nabla module.

This module provides JAX-like function transforms, starting with vmap
for automatic vectorization over batch dimensions.

vmap uses the nabla module's batch_dims mechanism:
1. Prepares inputs by moving specified axes into batch_dims
2. Calls the user function (ops work transparently with batch_dims)
3. Restores outputs by moving batch_dims back to specified positions

The axis specification supports JAX's "prefix pytree" semantics:
- Scalar (int/None) broadcasts to all tensor leaves
- Container must structurally match the corresponding input
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tensor import Tensor
    from ..sharding.mesh import DeviceMesh

# =============================================================================
# Type Definitions
# =============================================================================

# Recursive type for axis specifications (JAX-style prefix pytrees)
# - int: batch along this axis
# - None: broadcast (don't batch)
# - dict/list/tuple: per-element specification (must match input structure)
AxisSpec = Union[int, None, dict[str, "AxisSpec"], list["AxisSpec"], tuple["AxisSpec", ...]]

T = TypeVar("T")


# =============================================================================
# Axis Specification Utilities
# =============================================================================

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


# =============================================================================
# Prefix Pytree Operations
# =============================================================================

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
        return tree  # Non-tensor leaf passes through
    
    # Both are containers - recurse with matching structure
    if isinstance(tree, dict) and isinstance(prefix, dict):
        # Validate keys match
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
    
    return tree  # Non-tensor leaf


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


# =============================================================================
# Batch Size Validation
# =============================================================================

def _get_batch_size(tensor: Tensor, axis: AxisSpec):
    """Get batch dimension for a single tensor/axis pair.
    
    Returns the Dim object (StaticDim or SymbolicDim) at the specified axis.
    Returns None if axis is None.
    """
    if axis is None:
        return None
    if not isinstance(axis, int):
        raise TypeError(f"Expected int or None for axis at tensor leaf, got {type(axis).__name__}")
    
    shape = tensor.shape  # graph.Shape
    if shape.rank == 0:
        raise ValueError(
            f"Cannot batch scalar tensor (rank=0) along axis {axis}. "
            "Use in_axes=None to broadcast scalars."
        )
    normalized = _normalize_axis(axis, shape.rank)
    return shape[normalized]  # Returns Dim object


def _validate_batch_sizes(args: tuple, in_axes: tuple[AxisSpec, ...], axis_size: int | None):
    """Validate batch dimensions and return the common batch Dim.
    
    Returns:
        Dim object (StaticDim or SymbolicDim) representing the batch dimension.
    """
    from max.graph.dim import StaticDim
    
    dims = []
    for arg, ax in zip(args, in_axes):
        dims.extend(_collect_from_prefix(_get_batch_size, arg, ax))
    
    # Filter None values (from in_axes=None) 
    dims = [d for d in dims if d is not None]
    
    if not dims:
        # All axes are None - use axis_size or error
        if axis_size is not None:
            return StaticDim(axis_size)
        raise ValueError(
            "All in_axes are None (broadcast). Must specify axis_size "
            "to determine batch dimension size."
        )
    
    # Check consistency - Dim objects support equality!
    first = dims[0]
    if not all(d == first for d in dims):
        raise ValueError(f"Inconsistent batch dimensions along specified axes: {dims}")
    
    # If axis_size provided, must match
    if axis_size is not None:
        expected = StaticDim(axis_size)
        if first != expected:
            raise ValueError(f"axis_size={axis_size} doesn't match inferred batch dim {first}")
    
    return first  # Return the Dim (static or symbolic)


# =============================================================================
# Batching Primitives
# =============================================================================

def _batch_tensor(tensor: Tensor, axis: AxisSpec, batch_dim, spmd_axis_name: str | None, mesh: "DeviceMesh | None") -> Tensor:
    """Prepare tensor for batched execution by moving axis to batch_dims.
    
    The key insight: after adding/moving an axis to batch_dims, we need to 
    ensure it ends up at physical position 0 (the outermost batch position).
    This is critical for nested vmap to work correctly.
    
    Args:
        tensor: Input tensor
        axis: Axis specification (None for broadcast, int for batched axis)
        batch_dim: Dim object (StaticDim or SymbolicDim) for the batch dimension
        spmd_axis_name: Optional mesh axis name to shard the batch dimension on.
        mesh: Optional device mesh for sharding the batch dimension.
    """
    from ..ops import view as l_ops
    from ..ops import _physical as p_ops
    from ..ops import communication as comm_ops
    from max.graph.dim import StaticDim
    
    old_batch_dims = tensor._impl.batch_dims
    
    if axis is None:
        # Broadcast case: unsqueeze at LOGICAL axis 0, then broadcast
        # This adds a dimension at physical position old_batch_dims
        t = l_ops.unsqueeze(tensor, axis=0)  # LOGICAL unsqueeze
        
        # Build target LOGICAL shape with Dim object
        target_shape = (batch_dim,) + tuple(tensor.shape)
        
        # Broadcast if needed
        needs_broadcast = not (isinstance(batch_dim, StaticDim) and batch_dim.dim == 1)
        if needs_broadcast:
            t = l_ops.broadcast_to(t, shape=target_shape)  # LOGICAL broadcast
        
        # Now the new dim is at physical position old_batch_dims (front of logical)
        # Increment batch_dims - the new dim becomes the LAST batch dim
        t = p_ops.incr_batch_dims(t)
        
        # Move the last batch dim (at physical position old_batch_dims) to position 0
        if old_batch_dims > 0:
            t = p_ops.moveaxis(t, source=old_batch_dims, destination=0)
            
        # Apply sharding to the new batch axis (now at physical position 0)
        if spmd_axis_name is not None and mesh is not None:
             from ..sharding.spec import DimSpec
             
             # The axis to shard is at physical position 0.
             # We need to construct the full sharding spec for the tensor.
             # The tensor 't' currently has:
             # - physical batch dims: old_batch_dims + 1
             # - logical dims: len(t.shape)
             # Total physical rank: old_batch_dims + 1 + len(t.shape)
             
             physical_rank = old_batch_dims + 1 + len(t.shape)
             dim_specs = [DimSpec([]) for _ in range(physical_rank)]
             
             # Inherit existing sharding from input tensor (shifted by 1)
             # We must copy ALL specs (batch + logical) because broadcast preserves logical structure
             if tensor._impl.sharding:
                 for i in range(len(tensor._impl.sharding.dim_specs)):
                     if i + 1 < len(dim_specs):
                        dim_specs[i + 1] = tensor._impl.sharding.dim_specs[i].clone()
             
             # Set the new batch dimension (at index 0) to be sharded on spmd_axis_name
             dim_specs[0] = DimSpec([spmd_axis_name])
             
             t = comm_ops.shard_op(t, mesh, dim_specs)
             # print(f"DEBUG: _batch_tensor sharded t: {t} type: {type(t)}")
             # if t is None:
             #     print("DEBUG: shard_op returned None!")
    else:
        # Batched axis case: move LOGICAL axis to front of LOGICAL shape
        if axis != 0:
            # Normalize negative axis
            logical_rank = len(tensor.shape)
            norm_axis = axis if axis >= 0 else logical_rank + axis
            
            # Translate to physical axis
            physical_axis = old_batch_dims + norm_axis
            
            # Move to front of logical (= position old_batch_dims in physical)
            t = p_ops.moveaxis(tensor, source=physical_axis, destination=old_batch_dims)
        else:
            t = tensor
        
        # Now the batched axis is at physical position old_batch_dims (front of logical)
        # Apply sharding to the batch axis BEFORE incr_batch_dims (while it's still logical)
        if spmd_axis_name is not None and mesh is not None:
            from ..sharding.spec import DimSpec
            
            # The axis to shard is at physical position old_batch_dims (front of logical)
            # We need to create a PHYSICAL shard spec that:
            # 1. Inherits existing batch sharding for positions 0..old_batch_dims-1
            # 2. Shards the new batch axis (at old_batch_dims) on spmd_axis_name
            # 3. Leaves logical dims replicated
            
            logical_rank = len(t.shape)
            physical_rank = old_batch_dims + logical_rank
            
            # Start with replicated specs for all physical dims
            dim_specs = [DimSpec([]) for _ in range(physical_rank)]
            
            # Inherit existing batch sharding if tensor already has sharding
            if t._impl.sharding:
                for i in range(min(old_batch_dims, len(t._impl.sharding.dim_specs))):
                    dim_specs[i] = t._impl.sharding.dim_specs[i].clone()
            
            # Shard the new batch axis (at position old_batch_dims) on spmd_axis_name
            dim_specs[old_batch_dims] = DimSpec([spmd_axis_name])
            
            t = comm_ops.shard_op(t, mesh, dim_specs)
        
        # Increment batch_dims - it becomes the LAST batch dim
        t = p_ops.incr_batch_dims(t)
        
        # Move the last batch dim (at physical position old_batch_dims) to position 0
        if old_batch_dims > 0:
            t = p_ops.moveaxis(t, source=old_batch_dims, destination=0)

    return t


def _unbatch_tensor(tensor: Tensor, axis: AxisSpec, spmd_axis_name: str | None = None, mesh: "DeviceMesh | None" = None) -> Tensor:
    """Restore tensor after batched execution by moving batch_dims to axis.
    
    The reverse of _batch_tensor: moves the outermost batch dim (position 0)
    back to its original logical position.
    
    Note: spmd_axis_name sharding is PRESERVED on the output axis - we don't
    all_gather automatically. The user can explicitly gather if needed.
    """
    from ..ops import view as l_ops
    from ..ops import _physical as p_ops
    
    current_batch_dims = tensor._impl.batch_dims
    
    # First, move the front batch dim (position 0) to the last batch position
    # This reverses the final step of _batch_tensor
    if current_batch_dims > 1:
        # Move position 0 to position current_batch_dims - 1
        t = p_ops.moveaxis(tensor, source=0, destination=current_batch_dims - 1)
    else:
        t = tensor
    
    # Now decrement batch_dims - the last batch dim becomes front of logical
    t = p_ops.decr_batch_dims(t)
    
    # Note: We do NOT all_gather here - output stays sharded on spmd_axis_name
    # The user can explicitly all_gather if they need replicated output
    
    if axis is None:
        # Squeeze out the broadcast dimension at LOGICAL axis 0
        return l_ops.squeeze(t, axis=0)  # LOGICAL squeeze
    
    # Move front of logical to target LOGICAL axis position
    if axis != 0:
        new_batch_dims = t._impl.batch_dims
        logical_rank = len(t.shape)
        
        # Normalize negative axis
        norm_axis = axis if axis >= 0 else logical_rank + axis
        
        # The axis is currently at physical position new_batch_dims (front of logical)
        # Move it to physical position new_batch_dims + norm_axis
        source = new_batch_dims
        destination = new_batch_dims + norm_axis
        t = p_ops.moveaxis(t, source=source, destination=destination)
    
    return t


# =============================================================================
# Main Transform: vmap
# =============================================================================

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
    
    Args:
        func: Function to vectorize. If None, returns a decorator.
        in_axes: Axis specification for inputs:
            - int: Same axis for all inputs (default 0)
            - None: Broadcast all inputs (don't map)
            - tuple/list: Per-argument specification
            Each element can be a scalar (broadcasts to tensor leaves) or
            a pytree matching the argument's structure.
        out_axes: Axis specification for outputs (same format as in_axes).
        axis_size: Optional explicit batch size. Required when all in_axes
            are None (pure broadcast). If provided with batched inputs,
            must match the inferred batch size.
        spmd_axis_name: Optional mesh axis name to shard the batch dimension on.
        mesh: Device mesh for SPMD sharding. Required when spmd_axis_name is set.
    
    Returns:
        Vectorized function.
    
    Examples:
        >>> # Basic usage - maps over axis 0
        >>> @vmap
        ... def square(x): return x * x
        >>> square(Tensor.arange(0, 5))
        
        >>> # Batched + broadcast inputs
        >>> vmap(add, in_axes=(0, None))(batched_x, scalar_y)
        
        >>> # Different input/output axes
        >>> vmap(fn, in_axes=1, out_axes=2)(x)
        
        >>> # Pytree inputs with per-leaf axes
        >>> vmap(process, in_axes={'w': 0, 'b': None})(params)
        
        >>> # Pure broadcast with explicit axis_size
        >>> vmap(fn, in_axes=None, axis_size=10)(scalar)
        
        >>> # Nested vmap
        >>> vmap(vmap(fn))(x_with_two_batch_dims)
        
        >>> # Sharded data parallel execution
        >>> @vmap(spmd_axis_name="data")
        >>> def forward(x): ...
    """
    # Support decorator usage: @vmap or @vmap(in_axes=...)
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes, out_axes=out_axes, axis_size=axis_size, spmd_axis_name=spmd_axis_name, mesh=mesh)
    
    def vectorized(*args: Any) -> Any:
        if not args:
            raise ValueError("vmap requires at least one input argument.")
        
        # Broadcast in_axes to match positional args
        in_ax = _broadcast_to_args(in_axes, len(args))
        
        # Validate and get batch dimension (as Dim object)
        batch_dim = _validate_batch_sizes(args, in_ax, axis_size)
        
        # Batch all inputs
        batched = tuple(
            _map_prefix(_batch_tensor, arg, ax, batch_dim, spmd_axis_name, mesh)
            for arg, ax in zip(args, in_ax)
        )
        
        # Execute function with batched inputs
        outputs = func(*batched)
        
        # Normalize outputs for uniform processing
        is_single = not isinstance(outputs, (list, tuple))
        out_list = [outputs] if is_single else list(outputs)
        out_ax = _broadcast_to_args(out_axes, len(out_list))
        
        # Unbatch all outputs
        unbatched = [
            _map_prefix(_unbatch_tensor, out, ax, spmd_axis_name, mesh)
            for out, ax in zip(out_list, out_ax)
        ]
        
        # Return in original format
        return unbatched[0] if is_single else type(outputs)(unbatched)
    
    return vectorized


__all__ = ["vmap", "AxisSpec"]
