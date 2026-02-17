# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Comparison operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import (
    BinaryOperation,
    OpArgs,
    OpKwargs,
    OpResult,
    OpTensorValues,
    Operation,
    UnaryOperation,
)


class ComparisonOp(BinaryOperation):
    """Base for binary comparison operations."""

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
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
        self, args: list, kwargs: dict, output_sharding: Any = None
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

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.equal(args[0], args[1])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for comparison: None gradients (non-differentiable)."""
        return [None, None]


class NotEqualOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "not_equal"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.not_equal(args[0], args[1])]


class GreaterOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "greater"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.greater(args[0], args[1])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for comparison: None gradients (non-differentiable)."""
        return [None, None]


class GreaterEqualOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "greater_equal"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.greater_equal(args[0], args[1])]


class LessOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "less"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.greater(args[1], args[0])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for comparison: None gradients (non-differentiable)."""
        return [None, None]


class LessEqualOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "less_equal"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.greater_equal(args[1], args[0])]


class LogicalAndOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "logical_and"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.logical_and(args[0], args[1])]


class LogicalOrOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "logical_or"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.logical_or(args[0], args[1])]


class LogicalNotOp(ComparisonUnaryOp):
    @property
    def name(self) -> str:
        return "logical_not"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.logical_not(args[0])]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        return [None]


class LogicalXorOp(ComparisonOp):
    @property
    def name(self) -> str:
        return "logical_xor"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        return [ops.logical_xor(args[0], args[1])]


_equal_op = EqualOp()
_not_equal_op = NotEqualOp()
_greater_op = GreaterOp()
_greater_equal_op = GreaterEqualOp()
_less_op = LessOp()
_less_equal_op = LessEqualOp()
_logical_and_op = LogicalAndOp()
_logical_or_op = LogicalOrOp()
_logical_not_op = LogicalNotOp()
_logical_xor_op = LogicalXorOp()


def equal(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _equal_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def not_equal(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _not_equal_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def greater(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _greater_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def greater_equal(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _greater_equal_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def less(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _less_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def less_equal(x: Tensor, y: Tensor | float | int) -> Tensor:
    from .base import ensure_tensor

    return _less_equal_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def logical_and(x: Tensor, y: Tensor) -> Tensor:
    from .base import ensure_tensor

    return _logical_and_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def logical_or(x: Tensor, y: Tensor) -> Tensor:
    from .base import ensure_tensor

    return _logical_or_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


def logical_not(x: Tensor) -> Tensor:
    from .base import ensure_tensor

    return _logical_not_op([ensure_tensor(x)], {})[0]


def logical_xor(x: Tensor, y: Tensor) -> Tensor:
    from .base import ensure_tensor

    return _logical_xor_op([ensure_tensor(x), ensure_tensor(y)], {})[0]


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
    "logical_xor",
]
