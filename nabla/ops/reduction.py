# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Updated Reduction Operations
# ===----------------------------------------------------------------------=== #

"""Reduction ops using updated Operation base with auto batch_dims."""

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
        result = ops.sum(x, axis)
        if not keepdims:
            result = ops.squeeze(result, axis)
        return result
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        """Reduce: (d0, d1, ...) -> (d0, ...) with reduce_dim removed."""
        from ..sharding.propagation import reduce_template
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        return reduce_template(rank, [axis], keepdims).instantiate(input_shapes, output_shapes)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs: Any) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        # Normalize negative axis
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
        result = ops.mean(x, axis)
        if not keepdims:
            result = ops.squeeze(result, axis)
        return result
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        """Reduce: (d0, d1, ...) -> (d0, ...) with reduce_dim removed."""
        from ..sharding.propagation import reduce_template
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        return reduce_template(rank, [axis], keepdims).instantiate(input_shapes, output_shapes)
    
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
    return _reduce_sum_op(x, axis=axis, keepdims=keepdims)


def mean(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    return _mean_op(x, axis=axis, keepdims=keepdims)


__all__ = ["ReduceSumOp", "reduce_sum", "MeanOp", "mean"]
