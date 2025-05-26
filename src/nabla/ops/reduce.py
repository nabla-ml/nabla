"""Reduction operations."""

from typing import List, Union, Optional
import numpy as np
from max.driver import Tensor
from max.graph import ops, Value

from ..core.array import Array, Shape
from .operation import ReductionOperation


class SumOp(ReductionOperation):
    """Sum reduction operation."""

    def __init__(
        self,
        arg_shape: Shape,
        axes: Union[int, List[int], None] = None,
        keep_dims: bool = False,
    ):
        super().__init__("sum", axes, keep_dims)
        self.arg_shape = arg_shape
        self.axes = axes
        self.keep_dims = keep_dims

    def maxpr(self, args: List[Value], output: Array) -> None:
        axes = self.axes
        if axes is None:
            # Sum over all axes - iterate through each axis from the last to first
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

    def eagerxpr(self, args: List[Array], output: Array) -> None:
        np_result = np.sum(args[0].get_numpy(), axis=self.axes, keepdims=self.keep_dims)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: List[Array], cotangent: Array, output: Array
    ) -> List[Array]:
        from .view import broadcast_to

        return [broadcast_to(cotangent, self.arg_shape)]

    def jvp_rule(
        self, primals: List[Array], tangents: List[Array], output: Array
    ) -> Array:
        return sum(tangents[0], axes=self.axes, keep_dims=self.keep_dims)


def sum(
    arg: Array,
    axes: Optional[Union[int, List[int]]] = None,
    axis: Optional[Union[int, List[int]]] = None,
    keep_dims: bool = False,
) -> Array:
    """Sum array elements over given axes."""
    if axis is not None and axes is not None:
        raise ValueError("Cannot specify both 'axes' and 'axis' parameters")
    if axis is not None:
        axes = axis

    op = SumOp(arg.shape, axes, keep_dims)
    return op.forward(arg)
