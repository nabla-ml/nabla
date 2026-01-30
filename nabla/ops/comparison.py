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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for comparison: None gradients (non-differentiable)."""
        return (None, None)


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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for comparison: None gradients (non-differentiable)."""
        return (None, None)


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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for comparison: None gradients (non-differentiable)."""
        return (None, None)


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

    def physical_execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for LogicalNotOp."""
        from ..core import GRAPH
        from ..core.sharding import spmd

        mesh = spmd.get_mesh_from_args(args)

        with GRAPH.graph:
            shard_results = spmd.execute_on_shards(
                self.maxpr, args, kwargs, mesh, op=self
            )

        output_sharding, _, _ = spmd.infer_output_sharding(
            self, args, mesh, kwargs or {}
        )

        return (shard_results, output_sharding, mesh)


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
