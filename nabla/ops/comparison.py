# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comparison operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import BinaryOperation, Operation, UnaryOperation


class ComparisonOp(BinaryOperation):
    """Base for binary comparison operations."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from max.dtype import DType

        shapes, _, devices = super().compute_physical_shape(
            args, kwargs, output_sharding
        )
        num_shards = len(shapes)
        dtypes = [DType.bool] * num_shards
        return shapes, dtypes, devices


class ComparisonUnaryOp(UnaryOperation):
    """Base for unary comparison (logical) operations."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from max.dtype import DType

        shapes, _, devices = super().compute_physical_shape(
            args, kwargs, output_sharding
        )
        num_shards = len(shapes)
        dtypes = [DType.bool] * num_shards
        return shapes, dtypes, devices

if TYPE_CHECKING:
    pass


class EqualOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "equal"

    def kernel(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.equal(args[0], args[1])

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for comparison: None gradients (non-differentiable)."""
        return (None, None)


class NotEqualOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "not_equal"

    def kernel(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.not_equal(args[0], args[1])


class GreaterOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "greater"

    def kernel(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.greater(args[0], args[1])

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for comparison: None gradients (non-differentiable)."""
        return (None, None)


class GreaterEqualOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "greater_equal"

    def kernel(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.greater_equal(args[0], args[1])


class LessOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "less"

    def kernel(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.greater(args[1], args[0])

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for comparison: None gradients (non-differentiable)."""
        return (None, None)


class LessEqualOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "less_equal"

    def kernel(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.greater_equal(args[1], args[0])


class LogicalAndOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "logical_and"

    def kernel(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.logical_and(args[0], args[1])


class LogicalOrOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "logical_or"

    def kernel(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.logical_or(args[0], args[1])


class LogicalNotOp(ComparisonUnaryOp):
    @property
    def name(self) -> str:
        return "logical_not"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.logical_not(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        return (None,)


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
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_not",
]
