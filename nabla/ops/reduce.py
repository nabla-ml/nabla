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

"""Reduction operations."""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import TensorValue, ops

from ..core.tensor import Tensor, Shape
from .operation import ReductionOperation
from .view import squeeze, squeeze_batch_dims

# Public API
__all__ = ["sum", "sum_batch_dims", "mean", "max", "argmax"]


def _normalize_axes(
    axes: int | list[int] | tuple[int, ...] | None, ndim: int
) -> list[int]:
    """Normalize axes parameter to a list of integers."""
    if axes is None:
        return list(range(ndim))
    elif isinstance(axes, int):
        return [axes]
    elif isinstance(axes, (list, tuple)):
        return list(axes)
    else:
        raise TypeError(f"axes must be int, list, tuple, or None, got {type(axes)}")


class SumOp(ReductionOperation):
    """sum reduction operation."""

    def __init__(
        self,
        arg_shape: Shape,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(f"sum[axes={axes}]", axes, keep_dims=True)
        self.arg_shape = arg_shape
        self.axes = axes
        self.keep_dims = keep_dims

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output_symbol = args[0]

        # Normalize axes to handle None, int, or collections
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))

        for axis in normalized_axes:
            try:
                output_symbol = ops.sum(output_symbol, axis=axis) 
            except ValueError as e:
                raise ValueError(f"Failed to compute sum over axis {axis}: {e}. \nTry to update the Modular Package to the latest nightly via `pip uninstall -y modular && rm -rf ~/.cache/pip ~/.cache/realtec && pip install --pre modular --index-url https://dl.modular.com/public/nightly/python/simple/`. \nThis should fix this issue.")

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        if isinstance(self.axes, list):
            numpy_axes: int | tuple[int, ...] | None = tuple(self.axes)
        else:
            numpy_axes = self.axes

        np_result = np.sum(args[0].to_numpy(), axis=numpy_axes, keepdims=True)
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        if len(cotangent.shape) > len(primals[0].shape):
            return [cotangent]

        if output.shape != cotangent.shape:
            raise ValueError(
                f"In VJP rule for SumOp, "
                f"output shape {output.shape} "
                f"does not match cotangent shape {cotangent.shape}."
                f"primal shape: {primals[0].shape}, "
            )

        from .view import broadcast_to

        return [broadcast_to(cotangent, self.arg_shape)]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        return sum(tangents[0], axes=self.axes, keep_dims=True)


# noqa: A001 - Intentionally shadowing built-in 'sum' for API consistency
def sum(
    arg: Tensor,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Tensor:
    """Calculates the sum of tensor elements over given axes.

    This function reduces an tensor by summing its elements along the
    specified axes. If no axes are provided, the sum of all elements in the
    tensor is calculated.

    Parameters
    ----------
    arg : Tensor
        The input tensor to be summed.
    axes : int | list[int] | tuple[int, ...] | None, optional
        The axis or axes along which to perform the sum. If None (the
        default), the sum is performed over all axes, resulting in a scalar
        tensor.
    keep_dims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. This allows the result to broadcast
        correctly against the original tensor. Defaults to False.

    Returns
    -------
    Tensor
        An tensor containing the summed values.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([[1, 2, 3], [4, 5, 6]])

    Sum all elements:
    >>> nb.sum(x)
    Tensor([21], dtype=int32)

    Sum along an axis:
    >>> nb.sum(x, axes=0)
    Tensor([5, 7, 9], dtype=int32)

    Sum along an axis and keep dimensions:
    >>> nb.sum(x, axes=1, keep_dims=True)
    Tensor([[ 6],
           [15]], dtype=int32)
    """
    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        ndim = len(arg.shape)
        for axis in axes:
            if not -ndim <= axis < ndim:
                raise ValueError(
                    f"axis {axis} is out of bounds for tensor of dimension {ndim}"
                )

        axes = [axis if axis < 0 else axis - len(arg.shape) for axis in axes]

    else:
        axes = []
        for i in range(-len(arg.shape), 0):
            axes.append(i)

    axes = sorted(axes)
    op = SumOp(arg.shape, axes, keep_dims=keep_dims)
    res = op.forward(arg)

    if not keep_dims:
        # manually use the squeeze operation to squeeze remaining axes
        for axis in axes:
            res = squeeze(res, [axis])  # axes always negative

    return res


def mean(
    arg: Tensor,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Tensor:
    """Computes the arithmetic mean of tensor elements over given axes.

    This function calculates the average of an tensor's elements along the
    specified axes. If no axes are provided, the mean of all elements in the
    tensor is calculated.

    Parameters
    ----------
    arg : Tensor
        The input tensor for which to compute the mean.
    axes : int | list[int] | tuple[int, ...] | None, optional
        The axis or axes along which to compute the mean. If None (the default),
        the mean is computed over all axes, resulting in a scalar tensor.
    keep_dims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. This allows the result to broadcast
        correctly against the original tensor. Defaults to False.

    Returns
    -------
    Tensor
        An tensor containing the mean values, typically of a floating-point dtype.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([[1, 2, 3], [4, 5, 6]])

    Compute the mean of all elements:
    >>> nb.mean(x)
    Tensor([3.5], dtype=float32)

    Compute the mean along an axis:
    >>> nb.mean(x, axes=0)
    Tensor([2.5, 3.5, 4.5], dtype=float32)

    Compute the mean along an axis and keep dimensions:
    >>> nb.mean(x, axes=1, keep_dims=True)
    Tensor([[2.],
           [5.]], dtype=float32)
    """
    from .binary import div
    from .creation import tensor

    # First compute the sum
    sum_result = sum(arg, axes=axes, keep_dims=keep_dims)

    # Calculate the number of elements being averaged
    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        # Handle negative axes
        ndim = len(arg.shape)
        normalized_axes = []
        for axis in axes:
            if not -ndim <= axis < ndim:
                raise ValueError(
                    f"axis {axis} is out of bounds for tensor of dimension {ndim}"
                )
            if axis < 0:
                normalized_axes.append(len(arg.shape) + axis)
            else:
                normalized_axes.append(axis)

        # Count elements along reduced axes
        count = 1
        for axis in normalized_axes:
            if axis < len(arg.shape):
                count *= arg.shape[axis]
    else:
        # All axes - total number of elements
        count = 1
        for dim in arg.shape:
            count *= dim

    # Create count as a scalar tensor
    count_tensor = tensor(float(count), dtype=arg.dtype)

    # Divide sum by count
    return div(sum_result, count_tensor)


class SumBatchDimsOp(ReductionOperation):
    """sum reduction operation."""

    def __init__(
        self,
        arg_batch_dims: Shape,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(f"sum_batch_dims[axes={axes}]")
        self.arg_batch_dims = arg_batch_dims
        self.axes = axes
        self.keep_dims = keep_dims

    def compute_output_shape(self, *input_shapes):
        return input_shapes[0]

    def compute_output_batch_dims(self, *input_batch_dims):
        return self._compute_reduction_shape(input_batch_dims[0], self.axes)

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        # first we must subtract len(output.shape) from each axis value
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))
        axes = [ax - len(output.shape) for ax in normalized_axes]
        output_symbol = args[0]
        for axis in axes:
            try:
                output_symbol = ops.sum(output_symbol, axis=axis)
            except ValueError as e:
                raise ValueError(f"Failed to compute sum over batchdim axis {axis}: {e}. \nTry to update the Modular Package to the latest nightly via `pip uninstall -y modular && rm -rf ~/.cache/pip ~/.cache/realtec && pip install --pre modular --index-url https://dl.modular.com/public/nightly/python/simple/`. \nThis should fix this issue.")

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))
        axes = [ax - len(output.shape) for ax in normalized_axes]
        np_result = np.sum(
            args[0].to_numpy(), axis=tuple(axes) if axes else None, keepdims=True
        )
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .view import broadcast_batch_dims

        if len(cotangent.batch_dims) > len(primals[0].batch_dims):
            return [cotangent]

        if output.batch_dims != cotangent.batch_dims:
            raise ValueError(
                f"In VJP rule for SumBatchDimsOp, "
                f"output batch_dims {output.batch_dims} "
                f"do not match cotangent batch_dims {cotangent.batch_dims}."
                f"primal batch_dims: {primals[0].batch_dims}"
            )

        return [broadcast_batch_dims(cotangent, self.arg_batch_dims)]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        return sum_batch_dims(tangents[0], axes=self.axes, keep_dims=True)


def sum_batch_dims(
    arg: Tensor,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Tensor:
    """Calculates the sum of tensor elements over given batch dimension axes.

    This function is specialized for reducing batch dimensions, which are
    used in function transformations like `vmap`. It operates on the
    `batch_dims` of an tensor, leaving the standard `shape` unaffected.

    Parameters
    ----------
    arg : Tensor
        The input tensor with batch dimensions.
    axes : int | list[int] | tuple[int, ...] | None, optional
        The batch dimension axis or axes to sum over. If None, sums over all
        batch dimensions.
    keep_dims : bool, optional
        If True, the reduced batch axes are kept with size one. Defaults
        to False.

    Returns
    -------
    Tensor
        An tensor with specified batch dimensions reduced by the sum operation.
    """

    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        batch_dims_len = len(arg.batch_dims)
        for axis in axes:
            if not -batch_dims_len <= axis < batch_dims_len:
                raise ValueError(
                    f"axis {axis} is out of bounds for tensor with "
                    f"{batch_dims_len} batch dimensions"
                )

        axes = [axis if axis < 0 else axis - batch_dims_len for axis in axes]
    else:
        axes = []
        for i in range(-len(arg.batch_dims), 0):
            axes.append(i)

    axes = sorted(axes)
    op = SumBatchDimsOp(arg.batch_dims, axes, keep_dims)
    res = op.forward(arg)

    if not keep_dims:
        for axis in axes:
            res = squeeze_batch_dims(res, [axis])

    return res


class MaxOp(ReductionOperation):
    """Max reduction operation."""

    def __init__(
        self,
        arg_shape: Shape,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(f"max[axes={axes}]", axes, keep_dims=True)
        self.arg_shape = arg_shape
        self.axes = axes
        self.keep_dims = keep_dims

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output_symbol = args[0]

        # Normalize axes to handle None, int, or collections
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))

        for axis in normalized_axes:
            try:
                output_symbol = ops.max(output_symbol, axis=axis)
            except ValueError as e:
                raise ValueError(f"Failed to compute max over axis {axis}: {e}. \nTry to update the Modular Package to the latest nightly via `pip uninstall -y modular && rm -rf ~/.cache/pip ~/.cache/realtec && pip install --pre modular --index-url https://dl.modular.com/public/nightly/python/simple/`. \nThis should fix this issue.")

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        if isinstance(self.axes, list):
            numpy_axes: int | tuple[int, ...] | None = tuple(self.axes)
        else:
            numpy_axes = self.axes

        np_result = np.max(args[0].to_numpy(), axis=numpy_axes, keepdims=True)
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .binary import equal
        from .view import broadcast_to

        # Get the primal input
        primal = primals[0]

        # Broadcast cotangent to match primal shape
        cotangent_broadcasted = broadcast_to(cotangent, self.arg_shape)

        # Broadcast the output (max values) to match primal shape
        output_broadcasted = broadcast_to(output, self.arg_shape)

        # Create mask where primal equals the max value (output)
        mask = equal(primal, output_broadcasted)

        # Convert mask to float and multiply with broadcasted cotangent
        mask_float = mask.astype(primal.dtype)
        result = cotangent_broadcasted * mask_float

        return [result]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        from .binary import equal, mul
        from .view import broadcast_to

        # Create mask where input equals the max value
        primal = primals[0]
        max_result = max(primal, axes=self.axes, keep_dims=True)
        max_broadcasted = broadcast_to(max_result, self.arg_shape)
        mask = equal(primal, max_broadcasted)

        # Convert mask to float for arithmetic operations
        mask_float = mask.astype(primal.dtype)

        # Apply mask to tangents and sum over the reduced axes
        masked_tangents = mul(tangents[0], mask_float)
        return sum(masked_tangents, axes=self.axes, keep_dims=True)


def max(
    arg: Tensor,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Tensor:
    """Finds the maximum value of tensor elements over given axes.

    This function reduces an tensor by finding the maximum element along the
    specified axes. If no axes are provided, the maximum of all elements in the
    tensor is returned.

    Parameters
    ----------
    arg : Tensor
        The input tensor.
    axes : int | list[int] | tuple[int, ...] | None, optional
        The axis or axes along which to find the maximum. If None (the
        default), the maximum is found over all axes, resulting in a scalar
        tensor.
    keep_dims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. This allows the result to broadcast
        correctly against the original tensor. Defaults to False.

    Returns
    -------
    Tensor
        An tensor containing the maximum values.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([[1, 5, 2], [4, 3, 6]])

    Find the maximum of all elements:
    >>> nb.max(x)
    Tensor([6], dtype=int32)

    Find the maximum along an axis:
    >>> nb.max(x, axes=1)
    Tensor([5, 6], dtype=int32)

    Find the maximum along an axis and keep dimensions:
    >>> nb.max(x, axes=0, keep_dims=True)
    Tensor([[4, 5, 6]], dtype=int32)
    """
    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        ndim = len(arg.shape)
        for axis in axes:
            if not -ndim <= axis < ndim:
                raise ValueError(
                    f"axis {axis} is out of bounds for tensor of dimension {ndim}"
                )

        axes = [axis if axis < 0 else axis - len(arg.shape) for axis in axes]

    else:
        axes = []
        for i in range(-len(arg.shape), 0):
            axes.append(i)

    axes = sorted(axes)
    op = MaxOp(arg.shape, axes, keep_dims=keep_dims)
    res = op.forward(arg)

    if not keep_dims:
        # manually use the squeeze operation to squeeze remaining axes
        for axis in axes:
            res = squeeze(res, [axis])  # axes always negative

    return res


class ArgMaxOp(ReductionOperation):
    """
    ArgMax reduction operation. It is batch-aware and handles the physical axis.
    This Op internally behaves as if keep_dims=True.
    """

    def __init__(
        self,
        arg_shape: Shape,
        logical_axis: int | None,
    ):
        super().__init__(
            f"argmax[axis={logical_axis}]",
            axes=[logical_axis] if logical_axis is not None else None,
            keep_dims=True,
        )
        self.arg_shape = arg_shape
        self.logical_axis = logical_axis

    def compute_output_dtype(self, arg: Tensor) -> DType:
        return DType.int64

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        input_symbol = args[0]
        # physical_axis = self._get_physical_axis(output.batch_dims)

        if self.logical_axis is None:
            # Flatten everything except batch dims for reduction
            tmp_shape = output.batch_dims + (-1,)
            tmp_arg = ops.reshape(input_symbol, tmp_shape)
            result = ops.argmax(tmp_arg, axis=-1)
            res_shape = output.batch_dims + (1,) * len(self.arg_shape)
            output.tensor_value = ops.reshape(result, res_shape)
        else:
            # Assume that logical axes is always negative
            # output.tensor_value = ops.argmax(input_symbol, axis=self.logical_axis)
            # if output.device != _DEFAULT_CPU and (self.logical_axis != -1 or self.logical_axis != len(args[0].shape) - 1):
            #     input_symbol = ops.transpose(input_symbol, self.logical_axis, -1)
            #     result = ops.argmax(input_symbol, axis=-1)
            #     output.tensor_value = ops.transpose(result, -1, self.logical_axis)
            # else:
            try:
                output.tensor_value = ops.argmax(input_symbol, axis=self.logical_axis)
            except ValueError as e:
                raise ValueError(f"Failed to compute argmax over axis {self.logical_axis}: {e}. \nTry to update the Modular Package to the latest nightly via `pip uninstall -y modular && rm -rf ~/.cache/pip ~/.cache/realtec && pip install --pre modular --index-url https://dl.modular.com/public/nightly/python/simple/`. \nThis should fix this issue.")

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        primal = args[0].to_numpy()

        if self.logical_axis is None:
            tmp_shape = output.batch_dims + (-1,)
            tmp_arg = primal.reshape(tmp_shape)
            np_result = np.argmax(tmp_arg, axis=-1)
            res_shape = output.batch_dims + (1,) * len(self.arg_shape)
            res = np_result.reshape(res_shape)
            if res.ndim == 0:
                res = np.array(res)
            output.impl_(res)
        else:
            # Assume that logical axes is always negative
            res = np.argmax(primal, axis=self.logical_axis, keepdims=True)
            if res.ndim == 0:
                res = np.array(res)
            output.impl_(res)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .creation import zeros_like

        return [zeros_like(primals[0])]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        from .creation import zeros_like

        return zeros_like(output)


def argmax(
    arg: Tensor,
    axes: int | None = None,
    keep_dims: bool = False,
) -> Tensor:
    """Finds the indices of maximum tensor elements over a given axis.

    This function returns the indices of the maximum values along an axis. If
    multiple occurrences of the maximum value exist, the index of the first
    occurrence is returned.

    Parameters
    ----------
    arg : Tensor
        The input tensor.
    axes : int | None, optional
        The axis along which to find the indices of the maximum values. If
        None (the default), the tensor is flattened before finding the index
        of the overall maximum value.
    keep_dims : bool, optional
        If True, the axis which is reduced is left in the result as a
        dimension with size one. This is not supported when `axes` is None.
        Defaults to False.

    Returns
    -------
    Tensor
        An tensor of `int64` integers containing the indices of the maximum
        elements.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1, 5, 2, 5])
    >>> nb.argmax(x)
    Tensor(1, dtype=int64)

    >>> y = nb.tensor([[1, 5, 2], [4, 3, 6]])
    >>> nb.argmax(y, axes=1)
    Tensor([1, 2], dtype=int64)

    >>> nb.argmax(y, axes=0, keep_dims=True)
    Tensor([[1, 0, 1]], dtype=int64)
    """

    logical_axis: int | None
    ndim = len(arg.shape)

    # 1. Validate the user-provided 'axes' argument
    if axes is None:
        logical_axis = None
    elif isinstance(axes, int):
        if not -ndim <= axes < ndim:
            raise ValueError(
                f"axis {axes} is out of bounds for tensor of dimension {ndim}"
            )
        logical_axis = axes
    elif isinstance(axes, (list, tuple)):
        if len(axes) > 1:
            raise NotImplementedError("nabla.argmax does not support a tuple of axes.")
        if not axes:
            raise ValueError("axis must be an integer or None, not an empty sequence.")
        axis_val = axes[0]
        if not isinstance(axis_val, int):
            raise TypeError(
                f"axis must be an integer, but got {type(axis_val)} inside sequence."
            )
        if not -ndim <= axis_val < ndim:
            raise ValueError(
                f"axis {axis_val} is out of bounds for tensor of dimension {ndim}"
            )
        logical_axis = axis_val
    else:
        raise TypeError(f"Invalid type for axes: {type(axes)}")

    if arg.shape == () or np.prod(arg.shape) == 1:
        from .creation import zeros_like

        return zeros_like(arg).astype(DType.int64)

    # make axes always a negative value
    if logical_axis is not None and logical_axis >= 0:
        logical_axis = logical_axis - ndim

    if logical_axis is not None and arg.stage_realization:
        # If we are in JIT mode, we need to move the axis to the back
        from .view import move_axis_from_back, move_axis_to_back

        arg = move_axis_to_back(arg, logical_axis)
        op = ArgMaxOp(arg.shape, -1)
        res = op.forward(arg)
        # move the axis back to its original position
        res = move_axis_from_back(res, logical_axis)
    else:
        # If axes is None, we can directly use the ArgMaxOp with the original axis
        op = ArgMaxOp(arg.shape, logical_axis)
        res = op.forward(arg)

    # 3. Handle keep_dims
    if not keep_dims:
        if logical_axis is None:
            return res.reshape(())
        else:
            # Squeeze the original logical axis relative to the tensor's logical shape.
            squeeze_axis = logical_axis if logical_axis >= 0 else ndim + logical_axis
            res = squeeze(res, [squeeze_axis])

    return res