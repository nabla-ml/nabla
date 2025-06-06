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
from .view import squeeze, squeeze_batch_dims

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
        super().__init__(f"sum[axes={axes}]", axes, keep_dims=True)
        self.arg_shape = arg_shape
        self.axes = axes
        self.keep_dims = keep_dims

    def maxpr(self, args: list[Value], output: Array) -> None:
        axes = self.axes
        if axes is None:
            # Sum over all axes
            output_symbol = args[0]
            for axis in range(len(args[0].shape) - 1, -1, -1):
                output_symbol = ops.sum(output_symbol, axis=axis, keepdims=True)
        else:
            if isinstance(axes, int):
                axes = [axes]

            axes = sorted(axes, reverse=True)
            output_symbol = args[0]

            for axis in axes:
                output_symbol = ops.sum(output_symbol, axis=axis, keepdims=True)

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
                f"In VJP rule for ReduceSumOp, "
                f"output shape {output.shape} "
                f"does not match cotangent shape {cotangent.shape}."
                f"primal shape: {primals[0].shape}, "
            )
        
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
    op = ReduceSumOp(arg.shape, axes, keep_dims=keep_dims)
    res = op.forward(arg)

    # print("DEBUG sum:", res.shape, res.batch_dims, op.axes, keep_dims)

    if not keep_dims:
        # manually use the squeeze operation to squeeze remaining axes 
        for axis in sorted(op.axes, reverse=True):
            res = squeeze(res, [axis])
            
    return res


class SumBatchDimsOp(ReductionOperation):
    """sum reduction operation."""

    def __init__(
        self,
        arg_batch_dims: Shape,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(f"sum_batch_dims[axes={axes}]", axes, keep_dims=True)
        self.arg_batch_dims = arg_batch_dims
        self.axes = axes
        self.keep_dims = keep_dims

    def compute_output_shape(self, *input_shapes):
        return input_shapes[0]

    def compute_output_batch_dims(self, *input_batch_dims):
        return self._compute_reduction_shape(
            input_batch_dims[0], self.axes
        )

    def maxpr(self, args: list[Value], output: Array) -> None:
        axes = self.axes
        if isinstance(axes, int):
            axes = [axes]

        axes = sorted(axes, reverse=True)
        output_symbol = args[0]

        for axis in axes:
            output_symbol = ops.sum(output_symbol, axis=axis)

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

        np_result = np.sum(args[0].to_numpy(), axis=numpy_axes, keepdims=True)
        if np_result.ndim == 0:
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .view import broadcast_batch_dims

        # The VJP for sum_batch_dims needs to reverse the batch dimension reduction
        # The target should be the primal's batch_dims (before the sum was applied)
        # target_batch_dims = primals[0].batch_dims
        # cotangent_batch_dims = cotangent.batch_dims
        
        # # Handle dimension mismatches that occur with higher-order derivatives
        # result = cotangent
        
        # # If cotangent has more batch dimensions than target, we need to sum out extras
        # if len(cotangent_batch_dims) > len(target_batch_dims):
        #     # Calculate how many dimensions to sum out
        #     excess_dims = len(cotangent_batch_dims) - len(target_batch_dims)
            
        #     # Sum out the excess dimensions from the front (left side)
        #     # This handles the case where vmap has added extra batch dimensions
        #     for _ in range(excess_dims):
        #         result = sum_batch_dims(result, axes=[0], keep_dims=False)
                
        # # If cotangent batch_dims don't match target after summing, check compatibility
        # result_batch_dims = result.batch_dims
        
        # # Compare shapes element by element to identify any remaining mismatches
        # if len(result_batch_dims) == len(target_batch_dims):
        #     # Check if any dimensions are incompatible
        #     axes_to_sum = []
        #     for i, (result_dim, target_dim) in enumerate(zip(result_batch_dims, target_batch_dims)):
        #         # If dimensions don't match and can't be broadcast, we need to sum
        #         if result_dim != target_dim and result_dim != 1 and target_dim != 1:
        #             # This dimension needs to be summed out
        #             axes_to_sum.append(i)
            
        #     # Sum out incompatible dimensions
        #     if axes_to_sum:
        #         # Sum in reverse order to avoid index shifting
        #         for axis in reversed(axes_to_sum):
        #             result = sum_batch_dims(result, axes=[axis], keep_dims=False)
        
        # # Now broadcast to the target batch_dims
        # return [broadcast_batch_dims(result, target_batch_dims)]

        if len(cotangent.batch_dims) > len(primals[0].batch_dims):
            return [cotangent]
        
        if output.batch_dims != cotangent.batch_dims:
            raise ValueError(
                f"In VJP rule for SumBatchDimsOp, "
                f"output batch_dims {output.batch_dims} "
                f"do not match cotangent batch_dims {cotangent.batch_dims}."
                f"primal batch_dims: {primals[0].batch_dims}"
            )

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

        batch_dims_len = len(arg.batch_dims)
        axes = [axis if axis < 0 else axis - batch_dims_len for axis in axes]

    op = SumBatchDimsOp(arg.batch_dims, axes, keep_dims)
    res = op.forward(arg)

    # print("DEBUG sum_batch_dims:",res.batch_dims, res.shape, op.axes, keep_dims)
    if not keep_dims:
        # manually use the squeeze operation to squeeze remaining axes 
        for axis in sorted(op.axes, reverse=True):
            res = squeeze_batch_dims(res, [axis])
    return res
