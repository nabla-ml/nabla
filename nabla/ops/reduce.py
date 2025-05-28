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


class ReduceSumOp(ReductionOperation):
    """reduce_sum reduction operation."""

    def __init__(
        self,
        arg_shape: Shape,
        axes: int | list[int] | tuple[int, ...] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(f"reduce_sum[axes={axes}]", axes, keep_dims)
        self.arg_shape = arg_shape
        self.axes = axes
        self.keep_dims = keep_dims

    def maxpr(self, args: list[Value], output: Array) -> None:
        axes = self.axes
        if axes is None:
            # reduce_sum over all axes - iterate through each axis from the last to first
            output_symbol = args[0]
            for axis in range(len(args[0].shape) - 1, -1, -1):
                output_symbol = ops.sum(output_symbol, axis=axis)
                if not self.keep_dims:
                    output_symbol = ops.squeeze(output_symbol, axis=axis)
        else:
            if isinstance(axes, int):
                axes = [axes]

            # Sort axes in descending order to avoid index shifting issues
            axes = sorted(axes, reverse=True)
            output_symbol = args[0]

            for axis in axes:
                output_symbol = ops.sum(output_symbol, axis=axis)
                if not self.keep_dims:
                    output_symbol = ops.squeeze(output_symbol, axis=axis)

        output.tensor_value = output_symbol

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        # Convert axes list to tuple for numpy compatibility
        if isinstance(self.axes, list):
            numpy_axes: int | tuple[int, ...] | None = tuple(self.axes)
        else:
            numpy_axes = self.axes

        np_result = np.sum(
            args[0].get_numpy(), axis=numpy_axes, keepdims=self.keep_dims
        )
        # Ensure result is at least a 0-d array for DLPack conversion
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
        return reduce_sum(tangents[0], axes=self.axes, keep_dims=self.keep_dims)


def reduce_sum(
    arg: Array,
    axes: int | list[int] | tuple[int, ...] | None = None,
    keep_dims: bool = False,
) -> Array:
    """reduce_sum array elements over given axes."""
    # if axis is not None and axes is not None:
    #     raise ValueError("Cannot specify both 'axes' and 'axis' parameters")
    # if axis is not None:
    #     axes = axis

    op = ReduceSumOp(arg.shape, axes, keep_dims)
    return op.forward(arg)
