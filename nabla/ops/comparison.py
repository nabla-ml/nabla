# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comparison operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import BinaryOperation, Operation

if TYPE_CHECKING:
    pass


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


class LessOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "less"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.greater(args[1], args[0])


class LessEqualOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "less_equal"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.greater_equal(args[1], args[0])


class LogicalAndOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "logical_and"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.logical_and(args[0], args[1])


class LogicalOrOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "logical_or"

    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.logical_or(args[0], args[1])


class LogicalNotOp(Operation):
    @property
    def name(self) -> str:
        return "logical_not"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.logical_not(x)


equal = EqualOp()
not_equal = NotEqualOp()
greater = GreaterOp()
greater_equal = GreaterEqualOp()
less = LessOp()
less_equal = LessEqualOp()
logical_and = LogicalAndOp()
logical_or = LogicalOrOp()
logical_not = LogicalNotOp()

__all__ = [
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "logical_and", "logical_or", "logical_not",
]
