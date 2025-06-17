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
from max.driver import Tensor
from max.dtype import DType
from max.graph import TensorValue, ops

from ..core.array import Array, Shape
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

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output_symbol = args[0]

        # Normalize axes to handle None, int, or collections
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))

        for axis in normalized_axes:
            output_symbol = ops.sum(output_symbol, axis=axis)

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        if isinstance(self.axes, list):
            numpy_axes: int | tuple[int, ...] | None = tuple(self.axes)
        else:
            numpy_axes = self.axes

        np_result = np.sum(args[0].to_numpy(), axis=numpy_axes, keepdims=True)
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
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
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return sum(tangents[0], axes=self.axes, keep_dims=True)


# noqa: A001 - Intentionally shadowing built-in 'sum' for API consistency
def sum(
    arg: Array,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Array:
    """sum array elements over given axes."""
    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        axes = [axis if axis < 0 else axis - len(arg.shape) for axis in axes]

    else:
        axes = []
        for i in range(-len(arg.shape), 0):
            axes.append(i)

    sorted(axes)
    op = SumOp(arg.shape, axes, keep_dims=keep_dims)
    res = op.forward(arg)

    # print("DEBUG sum:", res.shape, res.batch_dims, op.axes, keep_dims)

    if not keep_dims:
        # manually use the squeeze operation to squeeze remaining axes
        for axis in axes:
            res = squeeze(res, [axis])  # axes always negative

    return res


def mean(
    arg: Array,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Array:
    """Compute mean of array elements over given axes."""
    from .binary import div
    from .creation import array

    # First compute the sum
    sum_result = sum(arg, axes=axes, keep_dims=keep_dims)

    # Calculate the number of elements being averaged
    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        # Handle negative axes
        normalized_axes = []
        for axis in axes:
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

    # Create count as a scalar array
    count_array = array([float(count)], dtype=arg.dtype)

    # Divide sum by count
    return div(sum_result, count_array)


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

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        # first we must subtract len(output.shape) from each axis value
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))
        axes = [ax - len(output.shape) for ax in normalized_axes]
        output_symbol = args[0]
        for axis in axes:
            output_symbol = ops.sum(output_symbol, axis=axis)

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))
        axes = [ax - len(output.shape) for ax in normalized_axes]
        np_result = np.sum(
            args[0].to_numpy(), axis=tuple(axes) if axes else None, keepdims=True
        )
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
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
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return sum_batch_dims(tangents[0], axes=self.axes, keep_dims=True)


def sum_batch_dims(
    arg: Array,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Array:
    """sum array elements over given batch dimension axes."""

    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        batch_dims_len = len(arg.batch_dims)
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

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output_symbol = args[0]

        # Normalize axes to handle None, int, or collections
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))

        for axis in normalized_axes:
            output_symbol = ops.max(output_symbol, axis=axis)

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        if isinstance(self.axes, list):
            numpy_axes: int | tuple[int, ...] | None = tuple(self.axes)
        else:
            numpy_axes = self.axes

        np_result = np.max(args[0].to_numpy(), axis=numpy_axes, keepdims=True)
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
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
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
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


class ArgMaxOp(ReductionOperation):
    """ArgMax reduction operation."""

    def __init__(
        self,
        arg_shape: Shape,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(f"argmax[axes={axes}]", axes, keep_dims=True)
        self.arg_shape = arg_shape
        self.axes = axes
        self.keep_dims = keep_dims

    def compute_output_dtype(self, arg: Array) -> DType:
        """ArgMax always returns integer indices."""
        return DType.int64

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output_symbol = args[0]

        # Apply argmax for each axis sequentially
        # Note: ops.argmax reduces along one axis and keeps the dimension (size 1)
        normalized_axes = _normalize_axes(self.axes, len(args[0].shape))
        for axis in normalized_axes:
            output_symbol = ops.argmax(output_symbol, axis=axis)

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        primal = args[0].to_numpy()

        # Normalize axes first
        normalized_axes = _normalize_axes(self.axes, primal.ndim)

        # Handle different cases for argmax
        if len(normalized_axes) == 1:
            # Single axis case
            axis = normalized_axes[0]
            np_result = np.argmax(primal, axis=axis, keepdims=True)
        elif len(normalized_axes) == len(primal.shape):
            # All axes case - flatten and argmax
            flat_array = primal.flatten()
            np_result = np.argmax(flat_array)
            np_result = np.array([[np_result]])  # Keep as 2D for consistency
        else:
            # Multiple specific axes - apply sequentially
            np_result = primal
            for axis in normalized_axes:
                np_result = np.argmax(np_result, axis=axis, keepdims=True)

        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        # ArgMax is not differentiable - return zero gradient
        from .creation import zeros_like

        return [zeros_like(primals[0])]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        # ArgMax is not differentiable - return zero tangent
        from .creation import zeros_like

        return zeros_like(output)


def max(
    arg: Array,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Array:
    """Find maximum array elements over given axes."""
    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        axes = [axis if axis < 0 else axis - len(arg.shape) for axis in axes]

    else:
        axes = []
        for i in range(-len(arg.shape), 0):
            axes.append(i)

    sorted(axes)
    op = MaxOp(arg.shape, axes, keep_dims=keep_dims)
    res = op.forward(arg)

    if not keep_dims:
        # manually use the squeeze operation to squeeze remaining axes
        for axis in axes:
            res = squeeze(res, [axis])  # axes always negative

    return res


def argmax(
    arg: Array,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Array:
    """Find indices of maximum array elements over given axes."""
    if axes is not None:
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, list | tuple):
            axes = [int(axis) for axis in axes]

        axes = [axis if axis < 0 else axis - len(arg.shape) for axis in axes]

    else:
        axes = []
        for i in range(-len(arg.shape), 0):
            axes.append(i)

    sorted(axes)
    op = ArgMaxOp(arg.shape, axes, keep_dims=keep_dims)
    res = op.forward(arg)

    if not keep_dims:
        # manually use the squeeze operation to squeeze remaining axes
        for axis in axes:
            res = squeeze(res, [axis])  # axes always negative

    return res
