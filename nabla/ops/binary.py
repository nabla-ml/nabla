# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.graph import TensorValue, ops

from ..core.array import Array
from .operation import BinaryOperation

# Public API
__all__ = [
    "add",
    "mul",
    "sub",
    "div",
    "floordiv",
    "pow",
    "greater_equal",
    "equal",
    "not_equal",
    "maximum",
    "minimum",
]


def _ensure_array(value) -> Array:
    """Convert scalar values to Arrays."""
    if isinstance(value, Array):
        return value
    elif isinstance(value, int | float):
        from .creation import array

        return array(value)
    else:
        raise TypeError(f"Cannot convert {type(value)} to Array")


class AddOp(BinaryOperation):
    """Addition operation."""

    def __init__(self):
        super().__init__("add")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.add(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.add(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [cotangent, cotangent]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return add(tangents[0], tangents[1])


class MulOp(BinaryOperation):
    """Multiplication operation."""

    def __init__(self):
        super().__init__("mul")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.mul(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.multiply(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [mul(cotangent, primals[1]), mul(cotangent, primals[0])]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return add(mul(primals[0], tangents[1]), mul(primals[1], tangents[0]))


class SubOp(BinaryOperation):
    """Subtraction operation."""

    def __init__(self):
        super().__init__("sub")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.sub(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.subtract(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .unary import negate

        return [cotangent, negate(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return sub(tangents[0], tangents[1])


class DivOp(BinaryOperation):
    """Division operation."""

    def __init__(self):
        super().__init__("div")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.div(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.divide(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .unary import negate

        x, y = primals
        cotangent_x = div(cotangent, y)
        cotangent_y = negate(div(mul(cotangent, x), mul(y, y)))
        return [cotangent_x, cotangent_y]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .unary import negate

        x, y = primals
        dx, dy = tangents
        term1 = div(dx, y)
        term2 = negate(div(mul(x, dy), mul(y, y)))
        return add(term1, term2)


class PowerOp(BinaryOperation):
    """Power operation (x^y)."""

    def __init__(self):
        super().__init__("pow")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.pow(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.pow(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .unary import log

        x, y = primals
        cotangent_x = mul(mul(cotangent, y), div(output, x))
        cotangent_y = mul(mul(cotangent, output), log(x))

        return [cotangent_x, cotangent_y]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .unary import log

        x, y = primals
        dx, dy = tangents
        term1 = mul(mul(y, div(output, x)), dx)
        term2 = mul(mul(output, log(x)), dy)

        return add(term1, term2)


class GreaterEqualOp(BinaryOperation):
    """Greater than or equal to operation."""

    def __init__(self):
        super().__init__("greater_equal")

    def compute_output_dtype(self, arg1: Array, arg2: Array) -> DType:
        """Comparison operations return bool dtype."""
        return DType.bool

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.greater_equal(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.greater_equal(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [0 * primals[0], 0 * primals[1]]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return greater_equal(tangents[0], tangents[1])


class MaximumOp(BinaryOperation):
    """Element-wise maximum operation."""

    def __init__(self):
        super().__init__("maximum")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.max(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.maximum(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        # Gradient flows to the larger input
        # For equal inputs, we split the gradient (JAX convention)
        x, y = primals
        x_greater = greater_equal(x, y)
        y_greater = greater_equal(y, x)

        # Cast boolean masks to float for multiplication
        from ..ops.unary import cast

        x_mask = cast(x_greater, cotangent.dtype)
        y_mask = cast(y_greater, cotangent.dtype)

        # When x == y, both masks are True, so we need to split the gradient
        both_equal = mul(x_mask, y_mask)
        x_only = sub(x_mask, both_equal)
        y_only = sub(y_mask, both_equal)

        # Split gradient equally when inputs are equal
        half_cotangent = mul(cotangent, 0.5)

        grad_x = add(mul(cotangent, x_only), mul(half_cotangent, both_equal))
        grad_y = add(mul(cotangent, y_only), mul(half_cotangent, both_equal))

        return [grad_x, grad_y]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        x, y = primals
        dx, dy = tangents
        x_greater = greater_equal(x, y)

        # Cast boolean mask to float for multiplication
        from ..ops.unary import cast

        x_mask = cast(x_greater, dx.dtype)
        y_mask = sub(1.0, x_mask)

        return add(mul(dx, x_mask), mul(dy, y_mask))


class MinimumOp(BinaryOperation):
    """Element-wise minimum operation."""

    def __init__(self):
        super().__init__("minimum")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.min(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.minimum(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        # Gradient flows to the smaller input
        # For equal inputs, we split the gradient (JAX convention)
        x, y = primals
        x_less_equal = greater_equal(y, x)  # x <= y
        y_less_equal = greater_equal(x, y)  # y <= x

        # Cast boolean masks to float for multiplication
        from ..ops.unary import cast

        x_mask = cast(x_less_equal, cotangent.dtype)
        y_mask = cast(y_less_equal, cotangent.dtype)

        # When x == y, both masks are True, so we need to split the gradient
        both_equal = mul(x_mask, y_mask)
        x_only = sub(x_mask, both_equal)
        y_only = sub(y_mask, both_equal)

        # Split gradient equally when inputs are equal
        half_cotangent = mul(cotangent, 0.5)

        grad_x = add(mul(cotangent, x_only), mul(half_cotangent, both_equal))
        grad_y = add(mul(cotangent, y_only), mul(half_cotangent, both_equal))

        return [grad_x, grad_y]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        x, y = primals
        dx, dy = tangents
        x_less_equal = greater_equal(y, x)  # x <= y

        # Cast boolean mask to float for multiplication
        from ..ops.unary import cast

        x_mask = cast(x_less_equal, dx.dtype)
        y_mask = sub(1.0, x_mask)

        return add(mul(dx, x_mask), mul(dy, y_mask))


class EqualOp(BinaryOperation):
    """Element-wise equality comparison operation."""

    def __init__(self):
        super().__init__("equal")

    def compute_output_dtype(self, arg0: Array, arg1: Array) -> DType:
        """Equal returns boolean dtype."""
        return DType.bool

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.equal(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        arg0_np = args[0].to_numpy()
        arg1_np = args[1].to_numpy()
        np_result = arg0_np == arg1_np
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .creation import zeros

        # Equal is not differentiable - return zero gradients
        zero_grad_0 = zeros(primals[0].shape, dtype=primals[0].dtype)
        zero_grad_1 = zeros(primals[1].shape, dtype=primals[1].dtype)
        return [zero_grad_0, zero_grad_1]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .creation import zeros

        # Equal is not differentiable
        return zeros(output.shape, dtype=output.dtype)


class NotEqualOp(BinaryOperation):
    """Element-wise not-equal comparison operation."""

    def __init__(self):
        super().__init__("not_equal")

    def compute_output_dtype(self, arg0: Array, arg1: Array) -> DType:
        """Not equal returns boolean dtype."""
        return DType.bool

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.not_equal(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        arg0_np = args[0].to_numpy()
        arg1_np = args[1].to_numpy()
        np_result = arg0_np != arg1_np
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .creation import zeros

        # Not equal is not differentiable - return zero gradients
        zero_grad_0 = zeros(primals[0].shape, dtype=primals[0].dtype)
        zero_grad_1 = zeros(primals[1].shape, dtype=primals[1].dtype)
        return [zero_grad_0, zero_grad_1]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .creation import zeros

        # Not equal is not differentiable
        return zeros(output.shape, dtype=output.dtype)


# Create operation instances
_add_op = AddOp()
_mul_op = MulOp()
_sub_op = SubOp()
_div_op = DivOp()
_power_op = PowerOp()
_greater_equal_op = GreaterEqualOp()
_maximum_op = MaximumOp()
_minimum_op = MinimumOp()
_equal_op = EqualOp()
_not_equal_op = NotEqualOp()


def add(arg0, arg1) -> Array:
    """Element-wise addition of two arrays or array and scalar."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _add_op.forward(arg0, arg1)


def mul(arg0, arg1) -> Array:
    """Element-wise multiplication of two arrays or array and scalar."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _mul_op.forward(arg0, arg1)


def sub(arg0, arg1) -> Array:
    """Element-wise subtraction of two arrays or array and scalar."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _sub_op.forward(arg0, arg1)


def div(arg0, arg1) -> Array:
    """Element-wise division of two arrays or array and scalar."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _div_op.forward(arg0, arg1)


def floordiv(arg0, arg1) -> Array:
    """Element-wise floor division of two arrays or array and scalar.

    Floor division is implemented as floor(a / b) which rounds towards
    negative infinity, matching Python's // operator behavior.
    """
    from ..ops.unary import floor

    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)

    # Perform regular division then floor
    result = div(arg0, arg1)
    return floor(result)


# noqa: A001 - Intentionally shadowing built-in 'pow' for API consistency
def pow(arg0, arg1) -> Array:
    """Element-wise power operation (arg0^arg1)."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _power_op.forward(arg0, arg1)


def greater_equal(arg0: Array, arg1: Array) -> Array:
    """Element-wise greater than or equal to operation."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _greater_equal_op.forward(arg0, arg1)


def maximum(arg0, arg1) -> Array:
    """Element-wise maximum of two arrays or array and scalar."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _maximum_op.forward(arg0, arg1)


def minimum(arg0, arg1) -> Array:
    """Element-wise minimum of two arrays or array and scalar."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _minimum_op.forward(arg0, arg1)


def equal(arg0, arg1) -> Array:
    """Element-wise equality comparison."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _equal_op.forward(arg0, arg1)


def not_equal(arg0, arg1) -> Array:
    """Element-wise not-equal comparison."""
    arg0 = _ensure_array(arg0)
    arg1 = _ensure_array(arg1)
    return _not_equal_op.forward(arg0, arg1)
