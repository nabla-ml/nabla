# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Updated Reduction Operations
# ===----------------------------------------------------------------------=== #

"""Reduction ops using updated Operation base with auto batch_dims."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


class MeanOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "mean"
    
    def maxpr(self, x: TensorValue, *, axis: int, keepdims: bool = False) -> TensorValue:
        result = ops.mean(x, axis)
        if not keepdims:
            result = ops.squeeze(result, axis)
        return result


_reduce_sum_op = ReduceSumOp()
_mean_op = MeanOp()


def reduce_sum(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    return _reduce_sum_op(x, axis=axis, keepdims=keepdims)


def mean(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    return _mean_op(x, axis=axis, keepdims=keepdims)


__all__ = ["ReduceSumOp", "reduce_sum", "MeanOp", "mean"]
