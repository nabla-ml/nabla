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

    _cost_multiplier = 4.0

    @property
    def name(self) -> str:
        return "sigmoid"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply sigmoid element-wise."""
        return ops.sigmoid(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))."""
        from ..ops.binary import mul, sub
        return mul(output, sub(1.0, output))


class TanhOp(UnaryOperation):
    """Hyperbolic tangent activation."""

    _cost_multiplier = 6.0

    @property
    def name(self) -> str:
        return "tanh"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply tanh element-wise."""
        return ops.tanh(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """tanh'(x) = 1 - tanh(x)^2."""
        from ..ops.binary import mul, sub
        return sub(1.0, mul(output, output))


class ExpOp(UnaryOperation):
    """Exponential function: e^x."""

    @property
    def name(self) -> str:
        return "exp"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply exp element-wise."""
        return ops.exp(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """exp'(x) = exp(x) = output."""
        return output


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

    _cost_multiplier = 3.0

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


class LogOp(UnaryOperation):
    """Natural logarithm: log(x)."""

    @property
    def name(self) -> str:
        return "log"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply log element-wise."""
        return ops.log(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """log'(x) = 1/x."""
        x = primals
        from ..ops.binary import div
        from ..ops.creation import ones_like
        return div(ones_like(x), x)


class SqrtOp(UnaryOperation):
    """Square root: sqrt(x)."""

    @property
    def name(self) -> str:
        return "sqrt"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply sqrt element-wise."""
        return ops.sqrt(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """sqrt'(x) = 1/(2*sqrt(x)) = 1/(2*output)."""
        from ..ops.binary import div, mul
        return div(1.0, mul(2.0, output))


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

    def _derivative(self, primals: Any, output: Any) -> Any:
        """acos'(x) = -1/sqrt(1 - x^2)."""
        x = primals
        from . import neg, sqrt
        from ..ops.binary import div, mul, sub
        return neg(div(1.0, sqrt(sub(1.0, mul(x, x)))))


class AtanhOp(UnaryOperation):
    """Inverse hyperbolic tangent: atanh(x)."""

    @property
    def name(self) -> str:
        return "atanh"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.atanh(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """atanh'(x) = 1/(1 - x^2)."""
        x = primals
        from ..ops.binary import div, mul, sub
        return div(1.0, sub(1.0, mul(x, x)))


class CosOp(UnaryOperation):
    """Cosine: cos(x)."""

    @property
    def name(self) -> str:
        return "cos"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.cos(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """cos'(x) = -sin(x)."""
        x = primals
        from . import neg, sin
        return neg(sin(x))


class ErfOp(UnaryOperation):
    """Error function: erf(x)."""

    @property
    def name(self) -> str:
        return "erf"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.erf(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """erf'(x) = (2/sqrt(pi)) * exp(-x^2)."""
        x = primals
        import math
        from . import exp, neg
        from ..ops.binary import mul
        factor = 2.0 / math.sqrt(math.pi)
        return mul(factor, exp(neg(mul(x, x))))


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


class _BoolOutputUnaryOp(UnaryOperation):
    """Base for unary ops that output bool dtype (is_inf, is_nan)."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from max.dtype import DType
        shapes, _, devices = super().compute_physical_shape(args, kwargs, output_sharding)
        return shapes, [DType.bool] * len(shapes), devices


class IsInfOp(_BoolOutputUnaryOp):
    """Check for infinity: is_inf(x)."""

    @property
    def name(self) -> str:
        return "is_inf"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.is_inf(x)


class IsNanOp(_BoolOutputUnaryOp):
    """Check for NaN: is_nan(x)."""

    @property
    def name(self) -> str:
        return "is_nan"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.is_nan(x)


class Log1pOp(UnaryOperation):
    """Log(1 + x): log1p(x)."""

    @property
    def name(self) -> str:
        return "log1p"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.log1p(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """log1p'(x) = 1/(1 + x)."""
        x = primals
        from ..ops.binary import add, div
        return div(1.0, add(1.0, x))


class RsqrtOp(UnaryOperation):
    """Reciprocal square root: 1 / sqrt(x)."""

    @property
    def name(self) -> str:
        return "rsqrt"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.rsqrt(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """rsqrt'(x) = -0.5 * rsqrt(x)^3 = -0.5 * output^3."""
        from ..ops.binary import mul
        return mul(-0.5, mul(output, mul(output, output)))


class SiluOp(UnaryOperation):
    """Sigmoid Linear Unit (swish): x * sigmoid(x)."""

    @property
    def name(self) -> str:
        return "silu"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.silu(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """silu'(x) = sigmoid(x) + silu(x) * (1 - sigmoid(x))."""
        x = primals
        from . import sigmoid
        from ..ops.binary import add, mul, sub
        sig_x = sigmoid(x)
        return add(sig_x, mul(output, sub(1.0, sig_x)))


class SinOp(UnaryOperation):
    """Sine: sin(x)."""

    @property
    def name(self) -> str:
        return "sin"

    def kernel(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.sin(x)

    def _derivative(self, primals: Any, output: Any) -> Any:
        """sin'(x) = cos(x)."""
        x = primals
        from . import cos
        return cos(x)


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

    def _derivative(self, primals: Any, output: Any) -> Any:
        """GELU'(x) = cdf(x) + x * pdf(x)."""
        x = primals
        import math
        from . import erf, exp, neg
        from ..ops.binary import add, div, mul
        sqrt2 = math.sqrt(2.0)
        sqrt2pi = math.sqrt(2.0 * math.pi)
        cdf = mul(0.5, add(1.0, erf(div(x, sqrt2))))
        pdf = div(exp(neg(div(mul(x, x), 2.0))), sqrt2pi)
        return add(cdf, mul(x, pdf))


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
        sum_st = reduce_sum(mul(soft, tangents), axis=axis, keepdims=True)
        return sub(tangents, sum_st)


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


def _is_reduction_axis_sharded(x: Tensor, axis: int) -> bool:
    """Check if the reduction axis is sharded across devices."""
    if not x.sharding:
        return False
    rank = len(x.shape)
    if axis < 0:
        axis += rank
    phys_axis = x.batch_dims + axis
    if phys_axis < len(x.sharding.dim_specs):
        spec = x.sharding.dim_specs[phys_axis]
        if spec.axes:
            return True
    return False


def logsoftmax(x: Tensor, axis: int = -1) -> Tensor:
    """LogSoftmax implementation with sharding support."""
    from ..ops.binary import sub
    from ..ops.reduction import reduce_max, reduce_sum
    from ..ops.unary import exp, log

    if _is_reduction_axis_sharded(x, axis):
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

    if _is_reduction_axis_sharded(x, axis):
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
