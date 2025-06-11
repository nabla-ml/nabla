from collections.abc import Callable
from typing import Any

from ..core.array import Array
from .utils import _handle_args_consistently


def _check_in_axes_size(tree: Any, axes: Any) -> int:
    """Check that all non-None axes have the same size and return that size.

    Args:
        tree: Pytree containing Arrays
        axes: Axis specification matching tree structure

    Returns:
        The common batch size for all non-None axes

    Raises:
        ValueError: If axes with non-None values have different sizes
    """
    batch_sizes = []

    def _collect_sizes(tree_part: Any, axes_part: Any) -> None:
        if isinstance(tree_part, Array):
            if axes_part is not None:
                # Handle scalar arrays (shape = ()) - they cannot be batched with a specific axis
                if len(tree_part.shape) == 0:
                    raise ValueError(
                        f"Cannot apply axis {axes_part} to scalar array with shape {tree_part.shape}. "
                        f"Scalar arrays cannot be batched along a specific axis."
                    )

                axis = len(tree_part.shape) + axes_part if axes_part < 0 else axes_part

                if axis >= len(tree_part.shape):
                    raise ValueError(
                        f"Axis {axes_part} out of bounds for array with shape {tree_part.shape}"
                    )

                batch_sizes.append(tree_part.shape[axis])
        elif isinstance(tree_part, dict):
            if isinstance(axes_part, dict):
                for k in tree_part:
                    _collect_sizes(tree_part[k], axes_part[k])
            else:
                # Broadcast axes_part to all dict values
                for k in tree_part:
                    _collect_sizes(tree_part[k], axes_part)
        elif isinstance(tree_part, list | tuple):
            if isinstance(axes_part, list | tuple):
                for t, a in zip(tree_part, axes_part, strict=False):
                    _collect_sizes(t, a)
            else:
                # Broadcast axes_part to all sequence elements
                for t in tree_part:
                    _collect_sizes(t, axes_part)
        # Non-Array leaves are ignored

    _collect_sizes(tree, axes)

    if not batch_sizes:
        # No non-None axes found, return 1 as default batch size
        return 1

    # Check all batch sizes are the same
    first_size = batch_sizes[0]
    for size in batch_sizes[1:]:
        if size != first_size:
            raise ValueError(
                f"Inconsistent batch sizes along specified axes: got sizes {batch_sizes}. "
                f"All non-None axes must have the same size."
            )

    return first_size


def _apply_batching_to_tree(
    tree: Any, axes: Any, is_input: bool = True, batch_size: int | None = None
) -> Any:
    """Apply batching/unbatching to a pytree structure.

    Args:
        tree: Pytree containing Arrays
        axes: Axis specification matching tree structure
        is_input: True for input batching, False for output unbatching
        batch_size: The batch size to use for broadcasting (axis=None case).
                   If None, uses size 1 for input batching.
    """

    def _process_array(array: Array, axis: int | None) -> Array:
        if is_input:
            # Input batching
            from nabla.ops.unary import incr_batch_dim_ctr
            from nabla.ops.view import unsqueeze

            if axis is None:
                # Broadcast: add batch dimension with correct size
                if batch_size is not None and batch_size > 1:
                    # Broadcast to the proper batch size
                    from nabla.ops.view import broadcast_to

                    # Add a size-1 dimension first
                    batched = unsqueeze(array, [0])
                    # Then broadcast to the correct batch size
                    new_shape = (batch_size,) + array.shape
                    batched = broadcast_to(batched, new_shape)
                else:
                    # Default behavior: add size-1 batch dimension
                    batched = unsqueeze(array, [0])
            else:
                # Move specified axis to position 0
                if axis != 0:
                    from ..ops.view import move_axis_to_front

                    batched = move_axis_to_front(array, axis)
                else:
                    batched = array

            res = incr_batch_dim_ctr(batched)

            from ..ops.view import move_axis_to_front_of_batch_dims

            return move_axis_to_front_of_batch_dims(res, -1)

        else:
            # Output unbatching
            from nabla.ops.unary import decr_batch_dim_ctr
            from nabla.ops.view import squeeze

            from ..ops.view import move_axis_from_front_of_batch_dims

            array = move_axis_from_front_of_batch_dims(array, -1)
            unbatched = decr_batch_dim_ctr(array)

            if axis is None:
                # Remove batch dimension
                unbatched = squeeze(unbatched, [0])
            else:
                # Move axis 0 to specified position
                if axis != 0:
                    from ..ops.view import move_axis_from_front

                    unbatched = move_axis_from_front(unbatched, axis)

            return unbatched

    def _recurse(tree_part: Any, axes_part: Any) -> Any:
        if isinstance(tree_part, Array):
            return _process_array(tree_part, axes_part)
        elif isinstance(tree_part, dict):
            if isinstance(axes_part, dict):
                return {k: _recurse(tree_part[k], axes_part[k]) for k in tree_part}
            else:
                # Broadcast axes_part to all dict values
                return {k: _recurse(tree_part[k], axes_part) for k in tree_part}
        elif isinstance(tree_part, list | tuple):
            if isinstance(axes_part, list | tuple):
                result = [
                    _recurse(t, a) for t, a in zip(tree_part, axes_part, strict=False)
                ]
                return type(tree_part)(result)
            else:
                # Broadcast axes_part to all sequence elements
                result = [_recurse(t, axes_part) for t in tree_part]
                return type(tree_part)(result)
        else:
            # Non-Array leaf, return unchanged
            return tree_part

    return _recurse(tree, axes)


def _broadcast_axis_spec(axis_spec: Any, num_items: int) -> tuple[Any, ...]:
    """Broadcast axis specification to match number of items."""
    if isinstance(axis_spec, int | type(None)):
        return tuple(axis_spec for _ in range(num_items))
    elif isinstance(axis_spec, list | tuple):
        if len(axis_spec) != num_items:
            raise ValueError(
                f"Axis specification length {len(axis_spec)} != number of items {num_items}"
            )
        return tuple(axis_spec)
    else:
        raise ValueError(f"Invalid axis specification: {axis_spec}")


def vmap(func=None, in_axes=0, out_axes=0) -> Callable[..., Any]:
    """Enhanced vmap with clean pytree support.

    This is a simplified, clean implementation that supports all JAX vmap features:
    - Pytree inputs/outputs with matching axis specifications
    - Broadcasting (axis=None) and batching (axis=int)
    - Nested structures (tuples, lists, dicts)
    - Both list-style and unpacked argument calling conventions
    """
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes, out_axes=out_axes)

    def vectorized_func(*args):
        # Handle calling conventions
        actual_args, is_list_style = _handle_args_consistently(args)

        if not actual_args:
            raise ValueError("vmap requires at least one input argument")

        # Broadcast in_axes to match arguments
        structured_in_axes = _broadcast_axis_spec(in_axes, len(actual_args))

        # Check that all non-None axes have the same size and get the batch size
        batch_size = _check_in_axes_size(actual_args, structured_in_axes)

        # Apply input batching with proper batch size
        batched_args = []
        for arg, axis_spec in zip(actual_args, structured_in_axes, strict=False):
            # Apply batching with the discovered batch size
            batched_arg = _apply_batching_to_tree(
                arg, axis_spec, is_input=True, batch_size=batch_size
            )
            batched_args.append(batched_arg)

        # Execute function
        outputs = func(batched_args) if is_list_style else func(*batched_args)

        # Handle output structure
        if not isinstance(outputs, list | tuple):
            outputs_list = [outputs]
            is_single_output = True
        else:
            outputs_list = outputs
            is_single_output = False

        # Broadcast out_axes to match outputs
        structured_out_axes = _broadcast_axis_spec(out_axes, len(outputs_list))

        # Apply output unbatching
        unbatched_outputs = []
        for output, axis_spec in zip(outputs_list, structured_out_axes, strict=False):
            unbatched_output = _apply_batching_to_tree(
                output, axis_spec, is_input=False
            )
            unbatched_outputs.append(unbatched_output)

        return unbatched_outputs[0] if is_single_output else tuple(unbatched_outputs)

    return vectorized_func
