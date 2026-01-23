# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import Operation, ReduceOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor

from .view import SqueezePhysicalOp

_squeeze_physical_op = SqueezePhysicalOp()


class ReduceSumOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_sum"

    def maxpr(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops.sum(x, axis)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for reduce_sum: broadcast cotangent back to input shape."""
        x = primals
        from ..ops.view.shape import broadcast_to
        return broadcast_to(cotangent, tuple(x.shape))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for reduce_sum: sum the tangents along the same axis."""
        t = tangents
        # Sum of tangents is the JVP
        axis = output.op_kwargs.get("axis", 0)
        return reduce_sum(t, axis=axis, keepdims=True)



class MeanOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "mean"

    def maxpr(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops.mean(x, axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for mean: broadcast cotangent / axis_size."""
        x = primals
        
        # Get axis from kwargs if available (from trace)
        axis = output.op_kwargs.get("axis", 0)
            
        axis_size = x.shape[axis]
        from ..ops.view.shape import broadcast_to
        # Create target shape for broadcasting cotangent back to x's shape
        target_shape = tuple(int(d) for d in x.shape)
        return broadcast_to(cotangent, target_shape) / axis_size

    def __call__(self, x, *, axis: int, keepdims: bool = False):
        return super().__call__(x, axis=axis, keepdims=keepdims)

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Mean: 1 sum + 1 div per output element."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return float(num_elements) + (
            float(num_elements) / input_shapes[0][0] if input_shapes[0] else 0
        )

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
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


class ReduceMaxOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_max"

    @property
    def collective_reduce_type(self) -> str:
        return "max"

    def maxpr(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops._reduce_max(x, axis=axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for reduce_max: one-hot mask where input == max."""
        x = primals
        from ..ops.comparison import equal
        from ..ops.view.shape import broadcast_to
        from ..ops.binary import mul
        # Broadcast output back to input shape for comparison
        max_broadcasted = broadcast_to(output, tuple(x.shape))
        mask = equal(x, max_broadcasted)  # 1.0 where x == max, 0.0 elsewhere
        cotangent_broadcasted = broadcast_to(cotangent, tuple(x.shape))
        return mul(cotangent_broadcasted, mask)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
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


class ReduceSumPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "reduce_sum_physical"

    def maxpr(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:

        return ops.sum(x, axis=axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce: (d0, d1, ...) -> (d0, 1, ...) with reduce_dim kept as size 1."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        out_factors = list(factors)
        if 0 <= axis < rank:
            out_factors[axis] = "1"
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs
    ) -> tuple[int, ...]:
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


class MeanPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "mean_physical"

    def maxpr(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:

        return ops.mean(x, axis=axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce: (d0, d1, ...) -> (d0, 1, ...) with reduce_dim kept as size 1."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        out_factors = list(factors)
        if 0 <= axis < rank:
            out_factors[axis] = "1"
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs
    ) -> tuple[int, ...]:
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


class ReduceMaxPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "reduce_max_physical"

    @property
    def collective_reduce_type(self) -> str:
        return "max"

    def maxpr(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:

        return ops._reduce_max(x, axis=axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce: (d0, d1, ...) -> (d0, 1, ...) with reduce_dim kept as size 1."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        out_factors = list(factors)
        if 0 <= axis < rank:
            out_factors[axis] = "1"
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs
    ) -> tuple[int, ...]:
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


class ReduceMinOp(ReduceOperation):
    @property
    def name(self) -> str:
        return "reduce_min"

    @property
    def collective_reduce_type(self) -> str:
        return "min"

    def maxpr(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops._reduce_min(x, axis=axis)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs: Any
    ) -> tuple[int, ...]:
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


class ReduceMinPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "reduce_min_physical"

    @property
    def collective_reduce_type(self) -> str:
        return "min"

    def maxpr(
        self, x: TensorValue, *, axis: int, keepdims: bool = False
    ) -> TensorValue:
        return ops._reduce_min(x, axis=axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce: (d0, d1, ...) -> (d0, 1, ...) with reduce_dim kept as size 1."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)

        out_factors = list(factors)
        if 0 <= axis < rank:
            out_factors[axis] = "1"
        out_str = " ".join(out_factors)

        return OpShardingRuleTemplate.parse(
            f"{in_str} -> {out_str}", input_shapes
        ).instantiate(input_shapes, output_shapes)

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs
    ) -> tuple[int, ...]:
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


_reduce_min_physical_op = ReduceMinPhysicalOp()
_reduce_min_op = ReduceMinOp()
_reduce_max_physical_op = ReduceMaxPhysicalOp()
_reduce_sum_physical_op = ReduceSumPhysicalOp()
_mean_physical_op = MeanPhysicalOp()
_reduce_sum_op = ReduceSumOp()
_mean_op = MeanOp()
_reduce_max_op = ReduceMaxOp()


def reduce_sum(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    from .view import squeeze

    result = _reduce_sum_op(x, axis=axis, keepdims=True)

    if not keepdims:
        result = squeeze(result, axis=axis)

    return result


def mean(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    """Compute arithmetic mean along specified axis.

    Implemented as sum(x) / shape[axis] to correctly handle distributed sharding.
    """
    s = reduce_sum(x, axis=axis, keepdims=keepdims)

    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis

    count = int(shape[axis])
    return s / count


def reduce_max(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    from .view import squeeze

    result = _reduce_max_op(x, axis=axis, keepdims=True)

    if not keepdims:
        result = squeeze(result, axis=axis)

    return result


def reduce_sum_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:

    result = _reduce_sum_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result


def mean_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:

    result = _mean_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result


def reduce_max_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:

    result = _reduce_max_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result


def reduce_min(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    from .view import squeeze

    result = _reduce_min_op(x, axis=axis, keepdims=True)

    if not keepdims:
        result = squeeze(result, axis=axis)

    return result


def reduce_min_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:

    result = _reduce_min_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result


__all__ = [
    "ReduceSumOp",
    "reduce_sum",
    "MeanOp",
    "mean",
    "ReduceMaxOp",
    "reduce_max",
    "ReduceSumPhysicalOp",
    "reduce_sum_physical",
    "MeanPhysicalOp",
    "mean_physical",
    "ReduceMaxPhysicalOp",
    "reduce_max_physical",
    "ReduceMinOp",
    "reduce_min",
    "ReduceMinPhysicalOp",
    "reduce_min_physical",
]
