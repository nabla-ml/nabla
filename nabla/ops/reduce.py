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
from max.graph import Value, ops

from ..core.array import Array, Shape
from .operation import ReductionOperation

# Public API
__all__ = ["sum", "sum_batch_dims"]


class ReduceSumOp(ReductionOperation):
    """sum reduction operation."""

    def __init__(
        self,
        arg_shape: Shape,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(f"sum[axes={axes}]", axes, keep_dims)
        self.arg_shape = arg_shape
        self.axes = axes
        self.keep_dims = keep_dims

    def maxpr(self, args: list[Value], output: Array) -> None:
        axes = self.axes
        # if axes is None:
        #     output_symbol = args[0]
        #     for axis in range(len(args[0].shape) - 1, -1, -1):
        #         output_symbol = ops.sum(output_symbol, axis=axis)
        #         if not self.keep_dims:
        #             output_symbol = ops.squeeze(output_symbol, axis=axis)
        # else:
        if isinstance(axes, int):
            axes = [axes]

        axes = sorted(axes, reverse=True)
        output_symbol = args[0]

        for axis in axes:
            output_symbol = ops.sum(output_symbol, axis=axis)
            if not self.keep_dims:
                output_symbol = ops.squeeze(output_symbol, axis=axis)

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        if isinstance(self.axes, list):
            numpy_axes: int | tuple[int, ...] | None = tuple(self.axes)
        else:
            numpy_axes = self.axes

        np_result = np.sum(args[0].to_numpy(), axis=numpy_axes, keepdims=self.keep_dims)
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .view import broadcast_to

        return [broadcast_to(cotangent, self.arg_shape)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return sum(tangents[0], axes=self.axes, keep_dims=self.keep_dims)


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

    op = ReduceSumOp(arg.shape, axes, keep_dims)
    return op.forward(arg)


class SumBatchDimsOp(ReductionOperation):
    """sum reduction operation."""

    def __init__(
        self,
        arg_batch_dims: Shape,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(f"sum_batch_dims[axes={axes}]", axes, keep_dims)
        self.arg_batch_dims = arg_batch_dims
        self.axes = axes
        self.keep_dims = keep_dims

    def compute_output_shape(self, *input_shapes):
        return input_shapes[0]

    def compute_output_batch_dims(self, *input_batch_dims):
        return self._compute_reduction_shape(
            input_batch_dims[0], self.axes, self.keep_dims
        )

    def maxpr(self, args: list[Value], output: Array) -> None:
        axes = self.axes
        # if axes is None:
        #     output_symbol = args[0]
        #     for axis in range(len(args[0].shape) - 1, -1, -1):
        #         output_symbol = ops.sum(output_symbol, axis=axis)
        #         if not self.keep_dims:
        #             output_symbol = ops.squeeze(output_symbol, axis=axis)
        # else:
        if isinstance(axes, int):
            axes = [axes]

        axes = sorted(axes, reverse=True)
        output_symbol = args[0]

        for axis in axes:
            output_symbol = ops.sum(output_symbol, axis=axis)
            if not self.keep_dims:
                output_symbol = ops.squeeze(output_symbol, axis=axis)

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        if isinstance(self.axes, list):
            numpy_axes: int | tuple[int, ...] | None = tuple(self.axes)
        else:
            numpy_axes = self.axes

        # SumBatchDimsOp operates on batch dimensions, which are the first dimensions of the array
        # The axes parameter is relative to batch_dims, not the full array
        # We need to convert batch_dims-relative axes to absolute array indices
        if numpy_axes is not None:
            if isinstance(numpy_axes, int):
                numpy_axes = [numpy_axes]
            elif isinstance(numpy_axes, list | tuple):
                numpy_axes = [int(axis) for axis in numpy_axes]

            # Convert batch_dims-relative indices to absolute array indices
            adjusted_axes = []
            batch_dims_len = len(args[0].batch_dims)

            for axis in numpy_axes:
                if axis < 0:
                    # Negative axis relative to batch_dims
                    # Convert to positive index within batch_dims range
                    adjusted_axis = batch_dims_len + axis
                else:
                    # Positive axis relative to batch_dims - use as is
                    adjusted_axis = axis

                # Validate the axis is within batch_dims range
                if adjusted_axis < 0 or adjusted_axis >= batch_dims_len:
                    raise IndexError(
                        f"Axis {axis} is out of bounds for batch_dims with length {batch_dims_len}"
                    )

                adjusted_axes.append(adjusted_axis)

            numpy_axes = tuple(adjusted_axes)

        np_result = np.sum(args[0].to_numpy(), axis=numpy_axes, keepdims=self.keep_dims)
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .view import broadcast_batch_dims

        return [broadcast_batch_dims(cotangent, self.arg_batch_dims)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return sum_batch_dims(tangents[0], axes=self.axes, keep_dims=self.keep_dims)


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

        # For batch dimensions, axes are relative to batch_dims
        # Normalize negative indices to positive indices within batch_dims range
        normalized_axes = []
        shape_len = len(arg.shape)
        batch_dims_len = len(arg.batch_dims)

        for axis in axes:
            # if axis < 0:
            #     # Convert negative index to positive within batch_dims range
            #     normalized_axis = batch_dims_len + axis
            # else:
            #     normalized_axis = axis

            # # Validate the axis is within batch_dims range
            # if normalized_axis < 0 or normalized_axis >= batch_dims_len:
            #     raise IndexError(f"Axis {axis} is out of bounds for batch_dims with length {batch_dims_len}")

            # normalized_axes.append(normalized_axis)
            if axis >= 0:
                # Convert positive axis to negative index relative to batch_dims
                normalized_axes.append(-batch_dims_len + axis)
            else:
                # Negative axis is already valid for batch_dims
                normalized_axes.append(axis)

        axes = normalized_axes

    op = SumBatchDimsOp(arg.batch_dims, axes, keep_dims)
    return op.forward(arg)
