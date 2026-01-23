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
        x = primals
        from ..ops.comparison import greater
        from ..ops.binary import mul
        # Derivative is 1 where x > 0, else 0
        mask = greater(x, 0.0)
        return mul(cotangent, mask)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for ReLU: tangent where x > 0, else 0."""
        x = primals
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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for sigmoid: ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x)) = output * (1 - output)."""
        from ..ops.binary import mul, sub
        # output = sigmoid(x), so ∂L/∂x = ∂L/∂output * output * (1 - output)
        one_minus_output = sub(1.0, output)
        sigmoid_grad = mul(output, one_minus_output)
        return mul(cotangent, sigmoid_grad)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for sigmoid: tangent * sigmoid(x) * (1 - sigmoid(x))."""
        t = tangents
        from ..ops.binary import mul, sub
        one_minus_output = sub(1.0, output)
        sigmoid_grad = mul(output, one_minus_output)
        return mul(t, sigmoid_grad)

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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for tanh: ∂tanh(x)/∂x = 1 - tanh(x)^2 = 1 - output^2."""
        from ..ops.binary import mul, sub
        # ∂L/∂x = ∂L/∂output * (1 - output^2)
        output_squared = mul(output, output)
        one_minus_output_sq = sub(1.0, output_squared)
        return mul(cotangent, one_minus_output_sq)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for tanh: tangent * (1 - tanh(x)^2)."""
        t = tangents
        from ..ops.binary import mul, sub
        output_squared = mul(output, output)
        one_minus_output_sq = sub(1.0, output_squared)
        return mul(t, one_minus_output_sq)

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
        """VJP for abs: grad = cotangent * sign(x)."""
        x = primals
        from ..ops.comparison import greater, less
        from ..ops.control_flow import where
        from ..ops.creation import ones_like, zeros_like
        from ..ops.binary import mul
        from . import neg
        
        # sign(x) = 1 if x > 0 else (-1 if x < 0 else 0)
        ones = ones_like(x)
        sign = where(greater(x, 0.0), ones, where(less(x, 0.0), neg(ones), zeros_like(x)))
        return mul(cotangent, sign)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for abs: tangent * sign(x)."""
        x = primals
        t = tangents
        from ..ops.comparison import greater, less
        from ..ops.control_flow import where
        from ..ops.creation import ones_like, zeros_like
        from ..ops.binary import mul
        from . import neg
        
        ones = ones_like(x)
        sign = where(greater(x, 0.0), ones, where(less(x, 0.0), neg(ones), zeros_like(x)))
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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for softmax: ∂s_i/∂x_j = s_i(δ_ij - s_j)."""
        # grad_x = output * (cotangent - sum(cotangent * output, axis, keepdims=True))
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum
        
        axis = output.op_kwargs.get("axis", -1)
        
        # Element-wise product of cotangent and softmax output
        cot_mul_out = mul(cotangent, output)
        
        # Sum along the softmax axis
        sum_cot_mul_out = reduce_sum(cot_mul_out, axis=axis, keepdims=True)
        
        # Final VJP logic
        return mul(output, sub(cotangent, sum_cot_mul_out))

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


class LogOp(UnaryOperation):
    """Natural logarithm: log(x)."""

    @property
    def name(self) -> str:
        return "log"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply log element-wise."""
        return ops.log(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for log: ∂log(x)/∂x = 1/x."""
        x = primals
        from ..ops.binary import div
        return div(cotangent, x)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for log: tangent / x."""
        x = primals
        t = tangents
        from ..ops.binary import div
        return div(t, x)


class SqrtOp(UnaryOperation):
    """Square root: sqrt(x)."""

    @property
    def name(self) -> str:
        return "sqrt"

    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply sqrt element-wise."""
        return ops.sqrt(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for sqrt: ∂sqrt(x)/∂x = 1/(2*sqrt(x)) = 1/(2*output)."""
        from ..ops.binary import div, mul
        # ∂L/∂x = ∂L/∂output * 1/(2*output)
        return div(cotangent, mul(2.0, output))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for sqrt: tangent / (2*sqrt(x))."""
        t = tangents
        from ..ops.binary import div, mul
        return div(t, mul(2.0, output))


relu = ReluOp()
sigmoid = SigmoidOp()
tanh = TanhOp()
exp = ExpOp()
neg = NegOp()
abs = AbsOp()
log = LogOp()
sqrt = SqrtOp()
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
    "LogOp",
    "SqrtOp",
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "neg",
    "abs",
    "log",
    "sqrt",
    "softmax",
]
