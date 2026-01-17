# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .operation import ReduceOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


class ReduceSumOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_sum"
    
    def maxpr(self, x: TensorValue, *, axis: int, keepdims: bool = False) -> TensorValue:
        return ops.sum(x, axis)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs: Any) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class MeanOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "mean"
    
    def maxpr(self, x: TensorValue, *, axis: int, keepdims: bool = False) -> TensorValue:
        return ops.mean(x, axis)
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Mean: 1 sum + 1 div per output element."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        # Sum cost + one division per output
        return float(num_elements) + (float(num_elements) / input_shapes[0][0] if input_shapes[0] else 0)
    
    # sharding_rule inherited from ReduceOperation
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs: Any) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


_reduce_sum_op = ReduceSumOp()
_mean_op = MeanOp()
def reduce_sum(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    from .view import squeeze
    
    # The sharding rule always assumes keepdims=True output shape then squeeze separately
    result = _reduce_sum_op(x, axis=axis, keepdims=True)
    
    if not keepdims:
        result = squeeze(result, axis=axis)
    
    return result


def mean(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    """Compute arithmetic mean along specified axis.
    
    Implemented as sum(x) / shape[axis] to correctly handle distributed sharding.
    """
    s = reduce_sum(x, axis=axis, keepdims=keepdims)
    
    # Get dimension size from global shape
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    
    count = int(shape[axis])
    return s / count


__all__ = ["ReduceSumOp", "reduce_sum", "MeanOp", "mean"]
