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

"""Function transforms for the eager module.

This module provides JAX-like function transforms, starting with vmap
for automatic vectorization over batch dimensions.

vmap uses the eager module's batch_dims mechanism:
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
    from .tensor import Tensor

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
    from .tensor import Tensor
    
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
    from .tensor import Tensor
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

def _get_batch_size(tensor: Tensor, axis: AxisSpec) -> int | None:
    """Get batch size for a single tensor/axis pair. Returns None if axis is None."""
    if axis is None:
        return None
    if not isinstance(axis, int):
        raise TypeError(f"Expected int or None for axis at tensor leaf, got {type(axis).__name__}")
    
    shape = tuple(tensor.shape)
    if not shape:
        raise ValueError(
            f"Cannot batch scalar tensor (shape=()) along axis {axis}. "
            "Use in_axes=None to broadcast scalars."
        )
    normalized = _normalize_axis(axis, len(shape))
    return shape[normalized]


def _validate_batch_sizes(args: tuple, in_axes: tuple[AxisSpec, ...], axis_size: int | None) -> int:
    """Validate batch sizes and return the common batch size."""
    sizes = []
    for arg, ax in zip(args, in_axes):
        sizes.extend(_collect_from_prefix(_get_batch_size, arg, ax))
    
    # Filter None values (from in_axes=None)
    sizes = [s for s in sizes if s is not None]
    
    if not sizes:
        # All axes are None - use axis_size or error
        if axis_size is not None:
            return axis_size
        raise ValueError(
            "All in_axes are None (broadcast). Must specify axis_size "
            "to determine batch dimension size."
        )
    
    # Check consistency
    first = sizes[0]
    if not all(s == first for s in sizes):
        raise ValueError(f"Inconsistent batch sizes along specified axes: {sizes}")
    
    # If axis_size provided, must match
    if axis_size is not None and axis_size != first:
        raise ValueError(f"axis_size={axis_size} doesn't match inferred batch size {first}")
    
    return first


# =============================================================================
# Batching Primitives
# =============================================================================

def _batch_tensor(tensor: Tensor, axis: AxisSpec, batch_size: int) -> Tensor:
    """Prepare tensor for batched execution by moving axis to batch_dims."""
    from . import view_ops
    
    if axis is None:
        # Broadcast: unsqueeze at front, optionally broadcast, mark as batch
        t = view_ops.unsqueeze(tensor, axis=0)
        if batch_size > 1:
            t = view_ops.broadcast_to(t, shape=(batch_size,) + tuple(tensor.shape))
        return view_ops.incr_batch_dims(t)
    
    # Move specified logical axis to batch_dims
    return view_ops.move_axis_to_batch_dims(tensor, axis=axis)


def _unbatch_tensor(tensor: Tensor, axis: AxisSpec) -> Tensor:
    """Restore tensor after batched execution by moving batch_dims to axis."""
    from . import view_ops
    
    if axis is None:
        # Squeeze out the broadcast dimension
        t = view_ops.decr_batch_dims(tensor)
        return view_ops.squeeze(t, axis=0)
    
    # Move outermost batch dim to specified logical position
    return view_ops.move_axis_from_batch_dims(tensor, batch_axis=0, logical_destination=axis)


# =============================================================================
# Main Transform: vmap
# =============================================================================

def vmap(
    func: Callable[..., T] | None = None,
    in_axes: AxisSpec = 0,
    out_axes: AxisSpec = 0,
    axis_size: int | None = None,
) -> Callable[..., T]:
    """Vectorize a function over batch dimensions.
    
    Creates a function that maps `func` over axes of its inputs, similar to
    JAX's vmap. Uses the eager module's batch_dims mechanism for transparent
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
    """
    # Support decorator usage: @vmap or @vmap(in_axes=...)
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes, out_axes=out_axes, axis_size=axis_size)
    
    def vectorized(*args: Any) -> Any:
        if not args:
            raise ValueError("vmap requires at least one input argument.")
        
        # Broadcast in_axes to match positional args
        in_ax = _broadcast_to_args(in_axes, len(args))
        
        # Validate and get batch size
        batch_size = _validate_batch_sizes(args, in_ax, axis_size)
        
        # Batch all inputs
        batched = tuple(
            _map_prefix(_batch_tensor, arg, ax, batch_size)
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
            _map_prefix(_unbatch_tensor, out, ax)
            for out, ax in zip(out_list, out_ax)
        ]
        
        # Return in original format
        return unbatched[0] if is_single else type(outputs)(unbatched)
    
    return vectorized


__all__ = ["vmap", "AxisSpec"]
