# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import LogicalAxisOperation, UnaryOperation

if TYPE_CHECKING:
    from ..core import Tensor


class ReluOp(UnaryOperation):
    """Rectified Linear Unit (ReLU) activation: max(0, x)."""

    @property
    def name(self) -> str:
        return "relu"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply ReLU element-wise."""
        return ops.relu(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for ReLU: ∂relu(x)/∂x = (x > 0)."""
        if isinstance(primals, tuple):
            x = primals[0]
        else:
            x = primals
        from ..ops.comparison import greater
        from ..ops.binary import mul
        # Derivative is 1 where x > 0, else 0
        mask = greater(x, 0.0)
        return mul(cotangent, mask)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for ReLU: tangent where x > 0, else 0."""
        if isinstance(primals, tuple):
            x = primals[0]
        else:
            x = primals
        if isinstance(tangents, tuple):
            t = tangents[0]
        else:
            t = tangents
        from ..ops.comparison import greater
        from ..ops.binary import mul
        mask = greater(x, 0.0)
        return mul(t, mask)


class SigmoidOp(UnaryOperation):
    """Sigmoid activation function: 1 / (1 + exp(-x))."""

    @property
    def name(self) -> str:
        return "sigmoid"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply sigmoid element-wise."""
        return ops.sigmoid(x)

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Sigmoid: ~4 FLOPs per element (neg, exp, add, div)."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return 4.0 * num_elements


class TanhOp(UnaryOperation):
    """Hyperbolic tangent activation."""

    @property
    def name(self) -> str:
        return "tanh"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply tanh element-wise."""
        return ops.tanh(x)

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Tanh: ~6 FLOPs per element (2 exp, 2 add/sub, 1 div)."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return 6.0 * num_elements


class ExpOp(UnaryOperation):
    """Exponential function: e^x."""

    @property
    def name(self) -> str:
        return "exp"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply exp element-wise."""
        return ops.exp(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for exp: ∂exp(x)/∂x = exp(x) = output."""
        from ..ops.binary import mul
        return mul(cotangent, output)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for exp: tangent * exp(x) = tangent * output."""
        if isinstance(tangents, tuple):
            t = tangents[0]
        else:
            t = tangents
        from ..ops.binary import mul
        return mul(output, t)


class NegOp(UnaryOperation):
    """Negation: -x."""

    @property
    def name(self) -> str:
        return "neg"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply negation element-wise."""
        return ops.negate(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for neg: ∂(-x)/∂x = -1."""
        return neg(cotangent)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for neg: -tangent."""
        if isinstance(tangents, tuple):
            t = tangents[0]
        else:
            t = tangents
        return neg(t)


class AbsOp(UnaryOperation):
    """Absolute value: |x|."""

    @property
    def name(self) -> str:
        return "abs"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply abs element-wise."""
        return ops.abs(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for abs: ∂|x|/∂x = sign(x)."""
        if isinstance(primals, tuple):
            x = primals[0]
        else:
            x = primals
        from ..ops.comparison import greater, less
        from ..ops.binary import mul, sub
        # sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
        pos_mask = greater(x, 0.0)
        neg_mask = less(x, 0.0)
        sign = sub(pos_mask, neg_mask)  # 1 - 0 = 1 for pos, 0 - 1 = -1 for neg
        return mul(cotangent, sign)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for abs: tangent * sign(x)."""
        if isinstance(primals, tuple):
            x = primals[0]
        else:
            x = primals
        if isinstance(tangents, tuple):
            t = tangents[0]
        else:
            t = tangents
        from ..ops.comparison import greater, less
        from ..ops.binary import mul, sub
        pos_mask = greater(x, 0.0)
        neg_mask = less(x, 0.0)
        sign = sub(pos_mask, neg_mask)
        return mul(t, sign)


class _SoftmaxNativeOp(LogicalAxisOperation, UnaryOperation):
    """Softmax activation function: exp(x) / sum(exp(x))."""

    @property
    def name(self) -> str:
        return "softmax"

    def __call__(self, x: Tensor, axis: int = -1) -> Tensor:
        """Apply softmax along specified axis."""
        return super().__call__(x, axis=axis)

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply softmax using MAX's native softmax."""
        axis = kwargs.get("axis", -1)
        return ops.softmax(x, axis=axis)

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Softmax: ~3 FLOPs per element (exp, sum, div)."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return 3.0 * num_elements


relu = ReluOp()
sigmoid = SigmoidOp()
tanh = TanhOp()
exp = ExpOp()
neg = NegOp()
abs = AbsOp()
_softmax_native = _SoftmaxNativeOp()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """A composition of existing nabla ops"""
    from ..ops.binary import div, sub
    from ..ops.reduction import reduce_max, reduce_sum
    from ..ops.unary import exp

    is_axis_sharded = False
    if x.sharding:
        rank = len(x.shape)
        if axis < 0:
            axis += rank

        phys_axis = x.batch_dims + axis
        if phys_axis < len(x.sharding.dim_specs):
            spec = x.sharding.dim_specs[phys_axis]
            if spec.axes:
                is_axis_sharded = True

    if is_axis_sharded:
        max_val = reduce_max(x, axis=axis, keepdims=True)
        shifted = sub(x, max_val)
        exp_val = exp(shifted)
        sum_val = reduce_sum(exp_val, axis=axis, keepdims=True)
        return div(exp_val, sum_val)

    return _softmax_native(x, axis=axis)


__all__ = [
    "ReluOp",
    "SigmoidOp",
    "TanhOp",
    "ExpOp",
    "NegOp",
    "AbsOp",
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "neg",
    "abs",
    "softmax",
]
