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

from ..core.tensor import Tensor
from .utils import (
    _handle_args_consistently,
    _map_pytree_with_axes,
)


def _check_in_axes_size(tree: Any, axes: Any) -> int:
    """Check that all non-None axes have the same size and return that size."""
    batch_sizes = []

    def _collect_sizes(tree_part: Any, axes_part: Any) -> None:
        if isinstance(tree_part, Tensor):
            if axes_part is not None:
                if len(tree_part.shape) == 0:
                    raise ValueError(
                        f"Cannot apply axis {axes_part} to scalar tensor with shape {tree_part.shape}. "
                        f"Scalar tensors cannot be batched along a specific axis."
                    )
                axis = len(tree_part.shape) + axes_part if axes_part < 0 else axes_part
                if axis >= len(tree_part.shape):
                    raise ValueError(
                        f"Axis {axes_part} out of bounds for tensor with shape {tree_part.shape}"
                    )
                batch_sizes.append(tree_part.shape[axis])
        elif isinstance(tree_part, dict):
            axes_map = axes_part if isinstance(axes_part, dict) else {k: axes_part for k in tree_part}
            for k, v in tree_part.items():
                _collect_sizes(v, axes_map.get(k))
        elif isinstance(tree_part, (list, tuple)):
            axes_list = axes_part if isinstance(axes_part, (list, tuple)) else [axes_part] * len(tree_part)
            for t, a in zip(tree_part, axes_list):
                _collect_sizes(t, a)

    _collect_sizes(tree, axes)

    if not batch_sizes:
        return 1

    first_size = batch_sizes[0]
    if not all(size == first_size for size in batch_sizes[1:]):
        raise ValueError(
            f"Inconsistent batch sizes along specified axes: got sizes {batch_sizes}. "
            f"All non-None axes must have the same size."
        )
    return first_size

def _batch_tensor(tensor: Tensor, axis: int | None, batch_size: int) -> Tensor:
    """Process a single tensor for batching in vmap."""
    from nabla.ops.unary import incr_batch_dim_ctr
    from nabla.ops.view import (
        broadcast_to,
        move_axis_to_front,
        move_axis_to_front_of_batch_dims,
        unsqueeze,
    )
    if axis is None:
        batched = unsqueeze(tensor, [0])
        if batch_size > 1:
            new_shape = (batch_size,) + tensor.shape
            batched = broadcast_to(batched, new_shape)
    else:
        batched = move_axis_to_front(tensor, axis) if axis != 0 else tensor
    
    res = incr_batch_dim_ctr(batched)
    return move_axis_to_front_of_batch_dims(res, -1)

def _unbatch_tensor(tensor: Tensor, axis: int | None) -> Tensor:
    """Process a single tensor for unbatching in vmap."""
    from nabla.ops.unary import decr_batch_dim_ctr
    from nabla.ops.view import (
        move_axis_from_front,
        move_axis_from_front_of_batch_dims,
        squeeze,
    )
    tensor = move_axis_from_front_of_batch_dims(tensor, -1)
    unbatched = decr_batch_dim_ctr(tensor)

    if axis is None:
        return squeeze(unbatched, [0])
    return move_axis_from_front(unbatched, axis) if axis != 0 else unbatched

def _batch_input_pytree(tree: Any, axes: Any, batch_size: int) -> Any:
    """Prepare a pytree of inputs for batched execution using the generic mapper."""
    return _map_pytree_with_axes(_batch_tensor, tree, axes, batch_size)

def _unbatch_output_pytree(tree: Any, axes: Any) -> Any:
    """Restore the original dimensions of a batched output pytree using the generic mapper."""
    return _map_pytree_with_axes(_unbatch_tensor, tree, axes)

def _broadcast_axis_spec(axis_spec: Any, num_items: int) -> tuple[Any, ...]:
    """Broadcast axis specification to match the number of pytree items."""
    if isinstance(axis_spec, (int, type(None))):
        return (axis_spec,) * num_items
    if isinstance(axis_spec, (list, tuple)):
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

    Parameters
    ----------
    func : Callable or None
        Function to vectorize
    in_axes : int or None or list or tuple, optional
        Specifies which axes to map over for inputs. Can be:
        - int: axis to map over (default 0)
        - None: broadcast (don't map)
        - list/tuple: per-input axis specification
    out_axes : int or None or list or tuple, optional
        Specifies which axes to map over for outputs (default 0)

    Returns
    -------
    Callable
        Vectorized function that maps func over the specified axes

    Examples
    --------
    >>> import nabla as nb
    >>> def square(x):
    ...     return x ** 2
    >>> x = nb.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> vmap_square = nb.vmap(square)
    >>> result = vmap_square(x)
    
    Multiple inputs with different axes:
    
    >>> def multiply(x, y):
    ...     return x * y
    >>> x = nb.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> y = nb.tensor([10.0, 20.0])
    >>> result = nb.vmap(multiply, in_axes=(0, None))(x, y)
    
    As a decorator:
    
    >>> @nb.vmap
    ... def process_batch(x):
    ...     return x ** 2 + 1
    >>> batch = nb.tensor([1.0, 2.0, 3.0, 4.0])
    >>> result = process_batch(batch)
    """
    if func is None:
        return lambda f: vmap(f, in_axes=in_axes, out_axes=out_axes)

    def vectorized_func(*args: Any) -> Any:
        actual_args, is_list_style = _handle_args_consistently(args)
        if not actual_args:
            raise ValueError("vmap requires at least one input argument.")

        structured_in_axes = _broadcast_axis_spec(in_axes, len(actual_args))
        batch_size = _check_in_axes_size(actual_args, structured_in_axes)

        batched_args = _batch_input_pytree(actual_args, structured_in_axes, batch_size)

        outputs = func(batched_args) if is_list_style else func(*batched_args)

        outputs_list, is_single_output = (
            ([outputs], True)
            if not isinstance(outputs, (list, tuple))
            else (list(outputs), False)
        )
        structured_out_axes = _broadcast_axis_spec(out_axes, len(outputs_list))
        unbatched_outputs = _unbatch_output_pytree(outputs_list, structured_out_axes)

        return unbatched_outputs[0] if is_single_output else tuple(unbatched_outputs)

    return vectorized_func


def xmap(
    func: Callable | None = None,
    in_axes: Union[int, None, list, tuple] = 0,
    out_axes: Union[int, None, list, tuple] = 0,
) -> Callable[..., Any]:
    pass