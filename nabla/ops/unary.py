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

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply ReLU element-wise."""
        return [ops.relu(args[0])]

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        """VJP for ReLU: ∂relu(x)/∂x = (x > 0)."""
        x = primals[0]
        from ..ops.comparison import greater
        from ..ops.control_flow import where
        from ..ops.creation import zeros_like

        # Derivative is 1 where x > 0, else 0
        mask = greater(x, 0.0)
        return [where(mask, cotangents[0], zeros_like(cotangents[0]))]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        """JVP for ReLU: tangent where x > 0, else 0."""
        x = primals[0]
        t = tangents[0]
        from ..ops.comparison import greater
        from ..ops.binary import mul

        mask = greater(x, 0.0)
        return [mul(t, mask)]


class SigmoidOp(UnaryOperation):
    """Sigmoid activation function: 1 / (1 + exp(-x))."""

    _cost_multiplier = 4.0

    @property
    def name(self) -> str:
        return "sigmoid"

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply sigmoid element-wise."""
        return [ops.sigmoid(args[0])]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply tanh element-wise."""
        return [ops.tanh(args[0])]

    def _derivative(self, primals: Any, output: Any) -> Any:
        """tanh'(x) = 1 - tanh(x)^2."""
        from ..ops.binary import mul, sub

        return sub(1.0, mul(output, output))


class ExpOp(UnaryOperation):
    """Exponential function: e^x."""

    @property
    def name(self) -> str:
        return "exp"

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply exp element-wise."""
        return [ops.exp(args[0])]

    def _derivative(self, primals: Any, output: Any) -> Any:
        """exp'(x) = exp(x) = output."""
        return output


class NegOp(UnaryOperation):
    """Negation: -x."""

    @property
    def name(self) -> str:
        return "neg"

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply negation element-wise."""
        return [ops.negate(args[0])]

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        """VJP for neg: ∂(-x)/∂x = -1."""
        return [neg(cotangents[0])]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        """JVP for neg: -tangent."""
        return [neg(tangents[0])]


class AbsOp(UnaryOperation):
    """Absolute value: |x|."""

    @property
    def name(self) -> str:
        return "abs"

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply abs element-wise."""
        return [ops.abs(args[0])]

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        """VJP for abs: grad = cotangent * sign(x)."""
        x = primals[0]
        cotangent = cotangents[0]
        from ..ops.comparison import greater, less
        from ..ops.control_flow import where
        from ..ops.creation import ones_like, zeros_like
        from ..ops.binary import mul
        from . import neg

        # grad = cotangent if x > 0 else (-cotangent if x < 0 else 0)
        from . import neg

        return [
            where(
                greater(x, 0.0),
                cotangent,
                where(less(x, 0.0), neg(cotangent), zeros_like(cotangent)),
            )
        ]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        """JVP for abs: tangent * sign(x)."""
        x = primals[0]
        t = tangents[0]
        from ..ops.comparison import greater, less
        from ..ops.control_flow import where
        from ..ops.creation import ones_like, zeros_like
        from ..ops.binary import mul
        from . import neg

        ones = ones_like(x)
        sign = where(
            greater(x, 0.0), ones, where(less(x, 0.0), neg(ones), zeros_like(x))
        )
        return [mul(t, sign)]


class _SoftmaxNativeOp(AxisOp, UnaryOperation):
    """Softmax activation function: exp(x) / sum(exp(x))."""

    _cost_multiplier = 3.0

    @property
    def name(self) -> str:
        return "softmax"

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply softmax using MAX's native softmax."""
        axis = kwargs.get("axis", -1)
        return [ops.softmax(args[0], axis=axis)]

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        """VJP for softmax: ∂s_i/∂x_j = s_i(δ_ij - s_j)."""
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum

        axis = kwargs.get("axis", -1)
        output = outputs[0]
        cotangent = cotangents[0]
        cot_mul_out = mul(cotangent, output)
        sum_cot_mul_out = reduce_sum(cot_mul_out, axis=axis, keepdims=True)
        return [mul(output, sub(cotangent, sum_cot_mul_out))]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum

        axis = kwargs.get("axis", -1)
        output = outputs[0]
        t_mul_out = mul(tangents[0], output)
        sum_t_mul_out = reduce_sum(t_mul_out, axis=axis, keepdims=True)
        return [mul(output, sub(tangents[0], sum_t_mul_out))]


class LogOp(UnaryOperation):
    """Natural logarithm: log(x)."""

    @property
    def name(self) -> str:
        return "log"

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply log element-wise."""
        return [ops.log(args[0])]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        """Apply sqrt element-wise."""
        return [ops.sqrt(args[0])]

    def _derivative(self, primals: Any, output: Any) -> Any:
        """sqrt'(x) = 1/(2*sqrt(x)) = 1/(2*output)."""
        from ..ops.binary import div, mul

        return div(1.0, mul(2.0, output))


_relu_op = ReluOp()
_sigmoid_op = SigmoidOp()
_tanh_op = TanhOp()
_exp_op = ExpOp()
_neg_op = NegOp()
_abs_op = AbsOp()
_log_op = LogOp()
_sqrt_op = SqrtOp()
_softmax_native = _SoftmaxNativeOp()


def relu(x: Tensor) -> Tensor:
    return _relu_op([x], {})[0]


def sigmoid(x: Tensor) -> Tensor:
    return _sigmoid_op([x], {})[0]


def tanh(x: Tensor) -> Tensor:
    return _tanh_op([x], {})[0]


def exp(x: Tensor) -> Tensor:
    return _exp_op([x], {})[0]


def neg(x: Tensor) -> Tensor:
    return _neg_op([x], {})[0]


def abs(x: Tensor) -> Tensor:
    return _abs_op([x], {})[0]


def log(x: Tensor) -> Tensor:
    return _log_op([x], {})[0]


def sqrt(x: Tensor) -> Tensor:
    return _sqrt_op([x], {})[0]


class AcosOp(UnaryOperation):
    """Arccosine: acos(x)."""

    @property
    def name(self) -> str:
        return "acos"

    def kernel(self, args: list, kwargs: dict) -> list:
        x = args[0]
        res = ops.acos(x)
        abs_x = ops.abs(x)
        one = ops.constant(1.0, x.type.dtype, x.type.device)
        mask = ops.greater(abs_x, one)
        nan = ops.constant(float("nan"), x.type.dtype, x.type.device)
        return [ops.where(mask, ops.broadcast_to(nan, x.type.shape), res)]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.atanh(args[0])]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.cos(args[0])]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.erf(args[0])]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.floor(args[0])]

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.creation import zeros_like

        return [zeros_like(cotangents[0])]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.creation import zeros_like

        return [zeros_like(outputs[0])]


class _BoolOutputUnaryOp(UnaryOperation):
    """Base for unary ops that output bool dtype (is_inf, is_nan)."""

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        from max.dtype import DType

        shapes, _, devices = super().compute_physical_shape(
            args, kwargs, output_sharding
        )
        return shapes, [DType.bool] * len(shapes), devices


class IsInfOp(_BoolOutputUnaryOp):
    """Check for infinity: is_inf(x)."""

    @property
    def name(self) -> str:
        return "is_inf"

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.is_inf(args[0])]


class IsNanOp(_BoolOutputUnaryOp):
    """Check for NaN: is_nan(x)."""

    @property
    def name(self) -> str:
        return "is_nan"

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.is_nan(args[0])]


class Log1pOp(UnaryOperation):
    """Log(1 + x): log1p(x)."""

    @property
    def name(self) -> str:
        return "log1p"

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.log1p(args[0])]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.rsqrt(args[0])]

    def _derivative(self, primals: Any, output: Any) -> Any:
        """rsqrt'(x) = -0.5 * rsqrt(x)^3 = -0.5 * output^3."""
        from ..ops.binary import mul

        return mul(-0.5, mul(output, mul(output, output)))


class SiluOp(UnaryOperation):
    """Sigmoid Linear Unit (swish): x * sigmoid(x)."""

    @property
    def name(self) -> str:
        return "silu"

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.silu(args[0])]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.sin(args[0])]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.trunc(args[0])]

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.creation import zeros_like

        return [zeros_like(cotangents[0])]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.creation import zeros_like

        return [zeros_like(outputs[0])]


class GeluOp(UnaryOperation):
    """Gaussian Error Linear Unit (GELU)."""

    @property
    def name(self) -> str:
        return "gelu"

    def kernel(self, args: list, kwargs: dict) -> list:
        x = args[0]
        approx = kwargs.get("approximate", "none")
        if approx is True:
            approx = "tanh"
        elif approx is False:
            approx = "none"
        return [ops.gelu(x, approximate=approx)]

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

    def kernel(self, args: list, kwargs: dict) -> list:
        axis = kwargs.get("axis", -1)
        return [ops.logsoftmax(args[0], axis=axis)]

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum

        axis = kwargs.get("axis", -1)
        output = outputs[0]
        cotangent = cotangents[0]
        soft = exp(output)
        sum_cot = reduce_sum(cotangent, axis=axis, keepdims=True)
        return [sub(cotangent, mul(soft, sum_cot))]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.binary import mul, sub
        from ..ops.reduction import reduce_sum

        axis = kwargs.get("axis", -1)
        output = outputs[0]
        soft = exp(output)
        sum_st = reduce_sum(mul(soft, tangents[0]), axis=axis, keepdims=True)
        return [sub(tangents[0], sum_st)]


class RoundOp(UnaryOperation):
    """Round to nearest integer: round(x)."""

    @property
    def name(self) -> str:
        return "round"

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.round(args[0])]

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.creation import zeros_like

        return [zeros_like(cotangents[0])]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        from ..ops.creation import zeros_like

        return [zeros_like(outputs[0])]


class CastOp(UnaryOperation):
    """Cast a tensor to a different data type."""

    @property
    def name(self) -> str:
        return "cast"

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.cast(args[0], kwargs["dtype"])]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        shapes, _, devices = super().compute_physical_shape(
            args, kwargs, output_sharding
        )
        dtype = kwargs.get("dtype")
        return shapes, [dtype] * len(shapes), devices

    def vjp_rule(
        self, primals: list, cotangents: list, outputs: list, kwargs: dict
    ) -> list:
        x = primals[0]
        return [cast(cotangents[0], dtype=x.dtype)]

    def jvp_rule(
        self, primals: list, tangents: list, outputs: list, kwargs: dict
    ) -> list:
        dtype = kwargs.get("dtype")
        return [cast(tangents[0], dtype=dtype)]


_acos_op = AcosOp()
_atanh_op = AtanhOp()
_cos_op = CosOp()
_erf_op = ErfOp()
_floor_op = FloorOp()
_is_inf_op = IsInfOp()
_is_nan_op = IsNanOp()
_log1p_op = Log1pOp()
_rsqrt_op = RsqrtOp()
_silu_op = SiluOp()
_sin_op = SinOp()
_trunc_op = TruncOp()
_gelu_op = GeluOp()
_round_op = RoundOp()
_cast_op = CastOp()
_logsoftmax_native = _LogSoftmaxNativeOp()


def acos(x: Tensor) -> Tensor:
    return _acos_op([x], {})[0]


def atanh(x: Tensor) -> Tensor:
    return _atanh_op([x], {})[0]


def cos(x: Tensor) -> Tensor:
    return _cos_op([x], {})[0]


def erf(x: Tensor) -> Tensor:
    return _erf_op([x], {})[0]


def floor(x: Tensor) -> Tensor:
    return _floor_op([x], {})[0]


def is_inf(x: Tensor) -> Tensor:
    return _is_inf_op([x], {})[0]


def is_nan(x: Tensor) -> Tensor:
    return _is_nan_op([x], {})[0]


def log1p(x: Tensor) -> Tensor:
    return _log1p_op([x], {})[0]


def rsqrt(x: Tensor) -> Tensor:
    return _rsqrt_op([x], {})[0]


def silu(x: Tensor) -> Tensor:
    return _silu_op([x], {})[0]


def sin(x: Tensor) -> Tensor:
    return _sin_op([x], {})[0]


def trunc(x: Tensor) -> Tensor:
    return _trunc_op([x], {})[0]


def gelu(x: Tensor) -> Tensor:
    return _gelu_op([x], {})[0]


def round(x: Tensor) -> Tensor:
    return _round_op([x], {})[0]


def cast(x: Tensor, dtype: Any = None) -> Tensor:
    return _cast_op([x], {"dtype": dtype})[0]


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

    return _logsoftmax_native([x], {"axis": axis})[0]


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

    return _softmax_native([x], {"axis": axis})[0]


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
