# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import AxisOp, UnaryOperation

if TYPE_CHECKING:
    from ..core import Tensor


class ReluOp(UnaryOperation):
    """Rectified Linear Unit (ReLU) activation: max(0, x)."""

    @property
    def name(self) -> str:
        return "relu"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply ReLU element-wise."""
        return ops.relu(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for ReLU: ∂relu(x)/∂x = (x > 0)."""
        x = primals
        from ..ops.comparison import greater
        from ..ops.control_flow import where
        from ..ops.creation import zeros_like

        # Derivative is 1 where x > 0, else 0
        mask = greater(x, 0.0)
        return where(mask, cotangent, zeros_like(cotangent))

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

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
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

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
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

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
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

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
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

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
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

        # grad = cotangent if x > 0 else (-cotangent if x < 0 else 0)
        from . import neg

        return where(
            greater(x, 0.0),
            cotangent,
            where(less(x, 0.0), neg(cotangent), zeros_like(cotangent)),
        )

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
        sign = where(
            greater(x, 0.0), ones, where(less(x, 0.0), neg(ones), zeros_like(x))
        )
        return mul(t, sign)


class _SoftmaxNativeOp(AxisOp, UnaryOperation):
    """Softmax activation function: exp(x) / sum(exp(x))."""

    @property
    def name(self) -> str:
        return "softmax"

    def __call__(self, x: Tensor, axis: int = -1) -> Tensor:
        """Apply softmax along specified axis."""
        return super().__call__(x, axis=axis)

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply softmax using MAX's native softmax."""
        axis = kwargs.get("axis", -1)
        return ops.softmax(x, axis=axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for softmax: ∂s_i/∂x_j = s_i(δ_ij - s_j)."""
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum

        axis = output.op_kwargs.get("axis", -1)
        cot_mul_out = mul(cotangent, output)
        sum_cot_mul_out = reduce_sum(cot_mul_out, axis=axis, keepdims=True)
        return mul(output, sub(cotangent, sum_cot_mul_out))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum

        axis = output.op_kwargs.get("axis", -1)
        t_mul_out = mul(tangents, output)
        sum_t_mul_out = reduce_sum(t_mul_out, axis=axis, keepdims=True)
        return mul(output, sub(tangents, sum_t_mul_out))

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

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
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

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
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


class AcosOp(UnaryOperation):
    """Arccosine: acos(x)."""

    @property
    def name(self) -> str:
        return "acos"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        res = ops.acos(x)
        # acos is only defined for |x| <= 1. MAX's ops.acos currently returns 0.0
        # for values outside this range, which is inconsistent with JAX and other
        # Nabla ops (like log, sqrt, atanh). We enforce NaN for correctness.
        abs_x = ops.abs(x)
        one = ops.constant(1.0, x.type.dtype, x.type.device)
        mask = ops.greater(abs_x, one)
        nan = ops.constant(float("nan"), x.type.dtype, x.type.device)
        return ops.where(mask, ops.broadcast_to(nan, x.type.shape), res)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: -cotangent / sqrt(1 - x^2)."""
        x = primals
        from . import neg, sqrt
        from ..ops.binary import div, mul, sub

        return neg(div(cotangent, sqrt(sub(1.0, mul(x, x)))))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: -tangent / sqrt(1 - x^2)."""
        x = primals
        t = tangents
        from . import neg, sqrt
        from ..ops.binary import div, mul, sub

        return neg(div(t, sqrt(sub(1.0, mul(x, x)))))


class AtanhOp(UnaryOperation):
    """Inverse hyperbolic tangent: atanh(x)."""

    @property
    def name(self) -> str:
        return "atanh"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.atanh(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: cotangent / (1 - x^2)."""
        x = primals
        from ..ops.binary import div, mul, sub

        return div(cotangent, sub(1.0, mul(x, x)))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: tangent / (1 - x^2)."""
        x = primals
        t = tangents
        from ..ops.binary import div, mul, sub

        return div(t, sub(1.0, mul(x, x)))


class CosOp(UnaryOperation):
    """Cosine: cos(x)."""

    @property
    def name(self) -> str:
        return "cos"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.cos(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: -cotangent * sin(x)."""
        x = primals
        from . import neg, sin
        from ..ops.binary import mul

        return neg(mul(cotangent, sin(x)))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: -tangent * sin(x)."""
        x = primals
        t = tangents
        from . import neg, sin
        from ..ops.binary import mul

        return neg(mul(t, sin(x)))


class ErfOp(UnaryOperation):
    """Error function: erf(x)."""

    @property
    def name(self) -> str:
        return "erf"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.erf(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: cotangent * (2 / sqrt(pi)) * exp(-x^2)."""
        x = primals
        import math

        from . import exp, neg
        from ..ops.binary import mul

        factor = 2.0 / math.sqrt(math.pi)
        return mul(cotangent, mul(factor, exp(neg(mul(x, x)))))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: tangent * (2 / sqrt(pi)) * exp(-x^2)."""
        x = primals
        t = tangents
        import math

        from . import exp, neg
        from ..ops.binary import mul

        factor = 2.0 / math.sqrt(math.pi)
        return mul(t, mul(factor, exp(neg(mul(x, x)))))


class FloorOp(UnaryOperation):
    """Floor (round down): floor(x)."""

    @property
    def name(self) -> str:
        return "floor"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.floor(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        from ..ops.creation import zeros_like

        return (zeros_like(cotangent),)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        from ..ops.creation import zeros_like

        return zeros_like(output)


class IsInfOp(UnaryOperation):
    """Check for infinity: is_inf(x)."""

    @property
    def name(self) -> str:
        return "is_inf"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.is_inf(x)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from max.dtype import DType

        shapes, _, devices = super().compute_physical_shape(
            args, kwargs, output_sharding
        )
        return shapes, [DType.bool] * len(shapes), devices


class IsNanOp(UnaryOperation):
    """Check for NaN: is_nan(x)."""

    @property
    def name(self) -> str:
        return "is_nan"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.is_nan(x)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from max.dtype import DType

        shapes, _, devices = super().compute_physical_shape(
            args, kwargs, output_sharding
        )
        return shapes, [DType.bool] * len(shapes), devices


class Log1pOp(UnaryOperation):
    """Log(1 + x): log1p(x)."""

    @property
    def name(self) -> str:
        return "log1p"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.log1p(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: cotangent / (1 + x)."""
        x = primals
        from ..ops.binary import add, div

        return div(cotangent, add(1.0, x))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: tangent / (1 + x)."""
        x = primals
        t = tangents
        from ..ops.binary import add, div

        return div(t, add(1.0, x))


class RsqrtOp(UnaryOperation):
    """Reciprocal square root: 1 / sqrt(x)."""

    @property
    def name(self) -> str:
        return "rsqrt"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.rsqrt(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: -0.5 * cotangent * output^3."""
        from ..ops.binary import mul

        return mul(-0.5, mul(cotangent, mul(output, mul(output, output))))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: -0.5 * tangent * output^3."""
        from ..ops.binary import mul

        return mul(-0.5, mul(tangents, mul(output, mul(output, output))))


class SiluOp(UnaryOperation):
    """Sigmoid Linear Unit (swish): x * sigmoid(x)."""

    @property
    def name(self) -> str:
        return "silu"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.silu(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: cotangent * (sigmoid(x) + output * (1 - sigmoid(x)))."""
        x = primals
        from . import sigmoid
        from ..ops.binary import add, mul, sub

        sig_x = sigmoid(x)
        return mul(cotangent, add(sig_x, mul(output, sub(1.0, sig_x))))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: tangent * (sigmoid(x) + output * (1 - sigmoid(x)))."""
        x = primals
        t = tangents
        from . import sigmoid
        from ..ops.binary import add, mul, sub

        sig_x = sigmoid(x)
        return mul(t, add(sig_x, mul(output, sub(1.0, sig_x))))


class SinOp(UnaryOperation):
    """Sine: sin(x)."""

    @property
    def name(self) -> str:
        return "sin"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.sin(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: cotangent * cos(x)."""
        x = primals
        from . import cos
        from ..ops.binary import mul

        return mul(cotangent, cos(x))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: tangent * cos(x)."""
        x = primals
        t = tangents
        from . import cos
        from ..ops.binary import mul

        return mul(t, cos(x))


class TruncOp(UnaryOperation):
    """Truncation (round towards zero): trunc(x)."""

    @property
    def name(self) -> str:
        return "trunc"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.trunc(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        from ..ops.creation import zeros_like

        return (zeros_like(cotangent),)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        from ..ops.creation import zeros_like

        return zeros_like(output)


class GeluOp(UnaryOperation):
    """Gaussian Error Linear Unit (GELU)."""

    @property
    def name(self) -> str:
        return "gelu"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        approx = kwargs.get("approximate", "none")
        if approx is True:
            approx = "tanh"
        elif approx is False:
            approx = "none"
        return ops.gelu(x, approximate=approx)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP: cotangent * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2 / 2) / sqrt(2 * pi))."""
        x = primals
        import math

        from . import erf, exp, neg
        from ..ops.binary import add, div, mul

        sqrt2 = math.sqrt(2.0)
        sqrt2pi = math.sqrt(2.0 * math.pi)

        cdf = mul(0.5, add(1.0, erf(div(x, sqrt2))))
        pdf = div(exp(neg(div(mul(x, x), 2.0))), sqrt2pi)

        deriv = add(cdf, mul(x, pdf))
        return mul(cotangent, deriv)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP: tangent * deriv."""
        x = primals
        t = tangents
        import math

        from . import erf, exp, neg
        from ..ops.binary import add, div, mul

        sqrt2 = math.sqrt(2.0)
        sqrt2pi = math.sqrt(2.0 * math.pi)

        cdf = mul(0.5, add(1.0, erf(div(x, sqrt2))))
        pdf = div(exp(neg(div(mul(x, x), 2.0))), sqrt2pi)

        deriv = add(cdf, mul(x, pdf))
        return mul(t, deriv)


class _LogSoftmaxNativeOp(AxisOp, UnaryOperation):
    """LogSoftmax activation function: log(exp(x) / sum(exp(x)))."""

    @property
    def name(self) -> str:
        return "logsoftmax"

    def __call__(self, x: Tensor, axis: int = -1) -> Tensor:
        return super().__call__(x, axis=axis)

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        axis = kwargs.get("axis", -1)
        return ops.logsoftmax(x, axis=axis)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum

        axis = output.op_kwargs.get("axis", -1)
        soft = exp(output)
        sum_cot = reduce_sum(cotangent, axis=axis, keepdims=True)
        return sub(cotangent, mul(soft, sum_cot))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum

        axis = output.op_kwargs.get("axis", -1)
        soft = exp(output)
        sum_t = reduce_sum(tangents, axis=axis, keepdims=True)
        return sub(tangents, mul(soft, sum_t))


class RoundOp(UnaryOperation):
    """Round to nearest integer: round(x)."""

    @property
    def name(self) -> str:
        return "round"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.round(x)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        from ..ops.creation import zeros_like

        return (zeros_like(cotangent),)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        from ..ops.creation import zeros_like

        return zeros_like(output)


class CastOp(UnaryOperation):
    """Cast a tensor to a different data type."""

    @property
    def name(self) -> str:
        return "cast"

    def kernel(self, x: TensorValue, *, dtype: DType) -> TensorValue:
        return ops.cast(x, dtype)

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        shapes, _, devices = super().compute_physical_shape(
            args, kwargs, output_sharding
        )
        dtype = kwargs.get("dtype")
        return shapes, [dtype] * len(shapes), devices

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        # primals = (x,)
        x = primals
        return (cast(cotangent, dtype=x.dtype),)

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        dtype = output.op_kwargs.get("dtype")
        return cast(tangents, dtype=dtype)


acos = AcosOp()
atanh = AtanhOp()
cos = CosOp()
erf = ErfOp()
floor = FloorOp()
is_inf = IsInfOp()
is_nan = IsNanOp()
log1p = Log1pOp()
rsqrt = RsqrtOp()
silu = SiluOp()
sin = SinOp()
trunc = TruncOp()
gelu = GeluOp()
round = RoundOp()
cast = CastOp()
_logsoftmax_native = _LogSoftmaxNativeOp()


def logsoftmax(x: Tensor, axis: int = -1) -> Tensor:
    """LogSoftmax implementation with sharding support."""
    from ..ops.binary import sub
    from ..ops.reduction import reduce_max, reduce_sum
    from ..ops.unary import exp, log

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
        return sub(shifted, log(sum_val))

    return _logsoftmax_native(x, axis=axis)


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
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "neg",
    "abs",
    "log",
    "sqrt",
    "acos",
    "atanh",
    "cos",
    "erf",
    "floor",
    "is_inf",
    "is_nan",
    "log1p",
    "rsqrt",
    "silu",
    "sin",
    "trunc",
    "gelu",
    "round",
    "cast",
    "softmax",
    "logsoftmax",
]
