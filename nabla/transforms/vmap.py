# ===----------------------------------------------------------------------=== #
# Nabla 2025
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

from collections.abc import Callable
from typing import Any, Union

from ..core.array import Array
from .utils import (
    _handle_args_consistently,
)


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


def _batch_input_pytree(tree: Any, axes: Any, batch_size: int) -> Any:
    """Prepare a pytree of inputs for batched execution.

    Moves the specified batch axis to the front for each array and
    broadcasts arrays where the axis is None.

    Args:
        tree: A pytree of Arrays.
        axes: A pytree of axis specifications.
        batch_size: The target batch size for broadcasting.

    Returns:
        A new pytree with batch dimensions moved to the front.
    """

    def _process_array(array: Array, axis: int | None) -> Array:
        from nabla.ops.unary import incr_batch_dim_ctr
        from nabla.ops.view import (
            broadcast_to,
            move_axis_to_front,
            move_axis_to_front_of_batch_dims,
            unsqueeze,
        )

        if axis is None:
            # Broadcast by adding a new axis and expanding it to the batch size.
            batched = unsqueeze(array, [0])
            if batch_size > 1:
                new_shape = (batch_size,) + array.shape
                batched = broadcast_to(batched, new_shape)
        else:
            # Move the existing batch axis to the front.
            batched = move_axis_to_front(array, axis) if axis != 0 else array

        # Increment batch dimension counter and finalize axis positioning.
        res = incr_batch_dim_ctr(batched)
        return move_axis_to_front_of_batch_dims(res, -1)

    def _recurse(tree_part: Any, axes_part: Any) -> Any:
        if isinstance(tree_part, Array):
            return _process_array(tree_part, axes_part)
        if isinstance(tree_part, dict):
            axes_map = (
                axes_part
                if isinstance(axes_part, dict)
                else dict.fromkeys(tree_part, axes_part)
            )
            return {k: _recurse(tree_part[k], axes_map[k]) for k in tree_part}
        if isinstance(tree_part, list | tuple):
            axes_list = (
                axes_part
                if isinstance(axes_part, list | tuple)
                else [axes_part] * len(tree_part)
            )
            return type(tree_part)(
                _recurse(t, a) for t, a in zip(tree_part, axes_list, strict=False)
            )
        return tree_part

    return _recurse(tree, axes)


def _unbatch_output_pytree(tree: Any, axes: Any) -> Any:
    """Restore the original dimensions of a batched output pytree.

    Moves the batch dimension from the front back to its specified
    output position or removes it if it was broadcasted.

    Args:
        tree: A pytree of batched Arrays.
        axes: A pytree of axis specifications for the output.

    Returns:
        A new pytree with batch dimensions restored to their original positions.
    """

    def _process_array(array: Array, axis: int | None) -> Array:
        from nabla.ops.unary import decr_batch_dim_ctr
        from nabla.ops.view import (
            move_axis_from_front,
            move_axis_from_front_of_batch_dims,
            squeeze,
        )

        # Reverse the batch dimension counter and initial axis movement.
        array = move_axis_from_front_of_batch_dims(array, -1)
        unbatched = decr_batch_dim_ctr(array)

        if axis is None:
            # Remove the broadcasted batch dimension.
            return squeeze(unbatched, [0])
        # Move the batch dimension to its final position.
        return move_axis_from_front(unbatched, axis) if axis != 0 else unbatched

    def _recurse(tree_part: Any, axes_part: Any) -> Any:
        if isinstance(tree_part, Array):
            return _process_array(tree_part, axes_part)
        if isinstance(tree_part, dict):
            axes_map = (
                axes_part
                if isinstance(axes_part, dict)
                else dict.fromkeys(tree_part, axes_part)
            )
            return {k: _recurse(tree_part[k], axes_map[k]) for k in tree_part}
        if isinstance(tree_part, list | tuple):
            axes_list = (
                axes_part
                if isinstance(axes_part, list | tuple)
                else [axes_part] * len(tree_part)
            )
            return type(tree_part)(
                _recurse(t, a) for t, a in zip(tree_part, axes_list, strict=False)
            )
        return tree_part

    return _recurse(tree, axes)


def _broadcast_axis_spec(axis_spec: Any, num_items: int) -> tuple[Any, ...]:
    """Broadcast axis specification to match the number of pytree items."""
    if isinstance(axis_spec, int | type(None)):
        return (axis_spec,) * num_items
    if isinstance(axis_spec, list | tuple):
        if len(axis_spec) != num_items:
            raise ValueError(
                f"Axis specification length {len(axis_spec)} does not match "
                f"number of items {num_items}"
            )
        return tuple(axis_spec)
    raise TypeError(f"Invalid axis specification type: {type(axis_spec)}")


def vmap(
    func: Callable | None = None,
    in_axes: Union[int, None, list, tuple] = 0,
    out_axes: Union[int, None, list, tuple] = 0,
) -> Callable[..., Any]:
    """Creates a function that maps a function over axes of pytrees.

    `vmap` is a transformation that converts a function designed for single
    data points into a function that can operate on batches of data points.
    It achieves this by adding a batch dimension to all operations within
    the function, enabling efficient, parallel execution.

    Args:
        func: The function to be vectorized. It should be written as if it
              operates on a single example.
        in_axes: Specifies which axis of the input(s) to map over. Can be an
                 integer, None, or a pytree of these values. `None` indicates
                 that the corresponding input should be broadcast.
        out_axes: Specifies where to place the batch axis in the output(s).

    Returns:
        A vectorized function with the same input/output structure as `func`.
    """
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes, out_axes=out_axes)

    def vectorized_func(*args: Any) -> Any:
        actual_args, is_list_style = _handle_args_consistently(args)
        if not actual_args:
            raise ValueError("vmap requires at least one input argument.")

        # 1. Prepare inputs and determine batch size
        structured_in_axes = _broadcast_axis_spec(in_axes, len(actual_args))
        batch_size = _check_in_axes_size(actual_args, structured_in_axes)

        # 2. Batch inputs by moving specified axes to the front
        batched_args = _batch_input_pytree(actual_args, structured_in_axes, batch_size)

        # 3. Execute the wrapped function with batched inputs
        outputs = func(batched_args) if is_list_style else func(*batched_args)

        # 4. Unbatch outputs by moving the batch axis to its final place
        outputs_list, is_single_output = (
            ([outputs], True)
            if not isinstance(outputs, list | tuple)
            else (list(outputs), False)
        )
        structured_out_axes = _broadcast_axis_spec(out_axes, len(outputs_list))
        unbatched_outputs = _unbatch_output_pytree(outputs_list, structured_out_axes)

        return unbatched_outputs[0] if is_single_output else tuple(unbatched_outputs)

    return vectorized_func
