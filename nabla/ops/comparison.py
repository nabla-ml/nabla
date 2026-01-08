# ===----------------------------------------------------------------------=== #
# Nabla 2026 - Comparison Operations
# ===----------------------------------------------------------------------=== #

"""Comparison operations."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .operation import BinaryOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


class EqualOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "equal"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.equal(args[0], args[1])


class NotEqualOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "not_equal"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.not_equal(args[0], args[1])


class GreaterOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "greater"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.greater(args[0], args[1])


class GreaterEqualOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "greater_equal"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.greater_equal(args[0], args[1])


# Implement Less/LessEqual via Greater swap to avoid missing primitives
class LessOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "less"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        # less(a, b) -> greater(b, a)
        return ops.greater(args[1], args[0])

class LessEqualOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "less_equal"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        # less_equal(a, b) -> greater_equal(b, a)
        return ops.greater_equal(args[1], args[0])


equal = EqualOp()
not_equal = NotEqualOp()
greater = GreaterOp()
greater_equal = GreaterEqualOp()
less = LessOp()
less_equal = LessEqualOp()

__all__ = [
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal"
]
