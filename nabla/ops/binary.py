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
from max.dtype import DType
from max.graph import TensorValue, ops

from ..core.tensor import Tensor
from .operation import BinaryOperation

# Public API
__all__ = [
    "add",
    "mul",
    "sub",
    "div",
    "floordiv",
    "mod",
    "pow",
    "greater_equal",
    "equal",
    "not_equal",
    "maximum",
    "minimum",
]


def _ensure_tensor(value) -> Tensor:
    """Convert scalar values to Tensors."""
    if isinstance(value, Tensor):
        return value
    elif isinstance(value, int | float):
        from .creation import tensor

        return tensor(value)
    else:
        raise TypeError(f"Cannot convert {type(value)} to Tensor")


class AddOp(BinaryOperation):
    """Addition operation."""

    def __init__(self):
        super().__init__("add")

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.add(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        np_result = np.add(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an tensor, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        return [cotangent, cotangent]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        return add(tangents[0], tangents[1])


class MulOp(BinaryOperation):
    """Multiplication operation."""

    def __init__(self):
        super().__init__("mul")

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.mul(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        np_result = np.multiply(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an tensor, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        return [mul(cotangent, primals[1]), mul(cotangent, primals[0])]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        return add(mul(primals[0], tangents[1]), mul(primals[1], tangents[0]))


class SubOp(BinaryOperation):
    """Subtraction operation."""

    def __init__(self):
        super().__init__("sub")

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.sub(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        np_result = np.subtract(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an tensor, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .unary import negate

        return [cotangent, negate(cotangent)]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        return sub(tangents[0], tangents[1])


class DivOp(BinaryOperation):
    """Division operation."""

    def __init__(self):
        super().__init__("div")

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.div(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        np_result = args[0].to_numpy() / args[1].to_numpy()
        # Ensure result is an tensor, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .unary import negate

        x, y = primals
        cotangent_x = div(cotangent, y)
        cotangent_y = negate(div(mul(cotangent, x), mul(y, y)))
        return [cotangent_x, cotangent_y]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
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

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.pow(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        np_result = args[0].to_numpy() ** args[1].to_numpy()
        # Ensure result is an tensor, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .unary import log

        x, y = primals
        cotangent_x = mul(mul(cotangent, y), div(output, x))
        cotangent_y = mul(mul(cotangent, output), log(x))

        return [cotangent_x, cotangent_y]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
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

    def compute_output_dtype(self, arg1: Tensor, arg2: Tensor) -> DType:
        """Comparison operations return bool dtype."""
        return DType.bool

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.greater_equal(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        import numpy as np

        np_result = np.greater_equal(args[0].to_numpy(), args[1].to_numpy())

        # Ensure result is always a numpy tensor
        if np.isscalar(np_result):
            np_result = np.array(np_result)

        # WORKAROUND: MAX library bug with scalar boolean tensors
        # The MAX tensor library fails when creating scalar boolean tensors
        # due to a bug in the _view method (line 49 in tensor.py)
        if np_result.shape == () and np_result.dtype == bool:
            # Convert scalar boolean to 1D boolean tensor, create tensor
            # The output will appear as scalar but be stored as 1D internally
            np_result_1d = np.array([np_result.item()], dtype=bool)
            output.impl_(np_result_1d)
            # Override the shape to appear as scalar
            output.shape = ()
        else:
            # Normal path for non-scalar boolean or any non-boolean results
            output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .creation import zeros_like

        return [
            zeros_like(cotangent).astype(primals[0].dtype),
            zeros_like(cotangent).astype(primals[1].dtype),
        ]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        from .creation import zeros_like

        return zeros_like(tangents[0]).astype(output.dtype)


class MaximumOp(BinaryOperation):
    """Element-wise maximum operation."""

    def __init__(self):
        super().__init__("maximum")

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.max(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        np_result = np.maximum(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an tensor, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
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
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
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

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.min(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        np_result = np.minimum(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an tensor, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
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
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
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

    def compute_output_dtype(self, arg0: Tensor, arg1: Tensor) -> DType:
        """Equal returns boolean dtype."""
        return DType.bool

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.equal(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        import numpy as np

        arg0_np = args[0].to_numpy()
        arg1_np = args[1].to_numpy()
        np_result = arg0_np == arg1_np

        # Ensure result is always a numpy tensor
        if np.isscalar(np_result):
            np_result = np.array(np_result)

        # WORKAROUND: MAX library bug with scalar boolean tensors
        # The MAX tensor library fails when creating scalar boolean tensors
        # Convert scalar boolean to float32 to avoid the bug
        if np_result.shape == () and np_result.dtype == bool:
            # Convert scalar boolean to float32 scalar (1.0 or 0.0)
            float_result = np_result.astype(np.float32)
            output.impl_(float_result)
            # Update output dtype to reflect what we actually stored
            output.dtype = DType.float32
        else:
            # Normal path for non-scalar boolean or any non-boolean results
            output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .creation import zeros_like

        return [
            zeros_like(cotangent).astype(primals[0].dtype),
            zeros_like(cotangent).astype(primals[1].dtype),
        ]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        from .creation import zeros_like

        return zeros_like(tangents[0])


class NotEqualOp(BinaryOperation):
    """Element-wise not-equal comparison operation."""

    def __init__(self):
        super().__init__("not_equal")

    def compute_output_dtype(self, arg0: Tensor, arg1: Tensor) -> DType:
        """Not equal returns boolean dtype."""
        return DType.bool

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.not_equal(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        import numpy as np

        arg0_np = args[0].to_numpy()
        arg1_np = args[1].to_numpy()
        np_result = arg0_np != arg1_np

        # Ensure result is always a numpy tensor
        if np.isscalar(np_result):
            np_result = np.array(np_result)

        # WORKAROUND: MAX library bug with scalar boolean tensors
        # The MAX tensor library fails when creating scalar boolean tensors
        # due to a bug in the _view method (line 49 in tensor.py)
        if np_result.shape == () and np_result.dtype == bool:
            # Convert scalar boolean to 1D boolean tensor, create tensor
            # The output will appear as scalar but be stored as 1D internally
            np_result_1d = np.array([np_result.item()], dtype=bool)
            output.impl_(np_result_1d)
            # Override the shape to appear as scalar
            output.shape = ()
        else:
            # Normal path for non-scalar boolean or any non-boolean results
            output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .creation import zeros_like

        return [
            zeros_like(cotangent).astype(primals[0].dtype),
            zeros_like(cotangent).astype(primals[1].dtype),
        ]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        from .creation import zeros_like

        return zeros_like(tangents[0]).astype(output.dtype)


class ModOp(BinaryOperation):
    """Modulo operation."""

    def __init__(self):
        super().__init__("mod")

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        output.tensor_value = ops.mod(args[0], args[1])

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        np_result = np.remainder(args[0].to_numpy(), args[1].to_numpy())
        # Ensure result is an tensor, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Tensor], cotangent: Tensor, output: Tensor
    ) -> list[Tensor]:
        from .unary import floor

        x, y = primals
        # For c = x % y = x - floor(x/y) * y
        # dc/dx = 1
        # dc/dy = -floor(x/y)
        cotangent_x = cotangent
        floor_div = floor(div(x, y))
        cotangent_y = mul(cotangent, mul(floor_div, -1))
        return [cotangent_x, cotangent_y]

    def jvp_rule(
        self, primals: list[Tensor], tangents: list[Tensor], output: Tensor
    ) -> Tensor:
        from .unary import floor

        x, y = primals
        dx, dy = tangents
        # For c = x % y = x - floor(x/y) * y
        # dc = dx - floor(x/y) * dy
        floor_div = floor(div(x, y))
        return sub(dx, mul(floor_div, dy))


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
_mod_op = ModOp()


def add(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Adds two tensors element-wise.

    This function performs element-wise addition on two tensors. It supports
    broadcasting, allowing tensors of different shapes to be combined as long
    as their shapes are compatible. This function also provides the
    implementation of the `+` operator for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar.
    y : Tensor | float | int
        The second input tensor or scalar. Must be broadcastable to the same
        shape as `x`.

    Returns
    -------
    Tensor
        An tensor containing the result of the element-wise addition.

    Examples
    --------
    Calling `add` explicitly:

    >>> import nabla as nb
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([4, 5, 6])
    >>> nb.add(x, y)
    Tensor([5, 7, 9], dtype=int32)

    Calling `add` via the `+` operator:

    >>> x + y
    Tensor([5, 7, 9], dtype=int32)

    Broadcasting a scalar:

    >>> x + 10
    Tensor([11, 12, 13], dtype=int32)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _add_op.forward(x, y)


def mul(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Multiplies two tensors element-wise.

    This function performs element-wise multiplication on two tensors. It
    supports broadcasting, allowing tensors of different shapes to be combined
    as long as their shapes are compatible. This function also provides the
    implementation of the `*` operator for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar.
    y : Tensor | float | int
        The second input tensor or scalar. Must be broadcastable to the same
        shape as `x`.

    Returns
    -------
    Tensor
        An tensor containing the result of the element-wise multiplication.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([4, 5, 6])
    >>> nb.mul(x, y)
    Tensor([4, 10, 18], dtype=int32)

    >>> x * y
    Tensor([4, 10, 18], dtype=int32)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _mul_op.forward(x, y)


def sub(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Subtracts two tensors element-wise.

    This function performs element-wise subtraction on two tensors. It supports
    broadcasting, allowing tensors of different shapes to be combined as long
    as their shapes are compatible. This function also provides the
    implementation of the `-` operator for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar (the minuend).
    y : Tensor | float | int
        The second input tensor or scalar (the subtrahend). Must be
        broadcastable to the same shape as `x`.

    Returns
    -------
    Tensor
        An tensor containing the result of the element-wise subtraction.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([10, 20, 30])
    >>> y = nb.tensor([1, 2, 3])
    >>> nb.sub(x, y)
    Tensor([ 9, 18, 27], dtype=int32)

    >>> x - y
    Tensor([ 9, 18, 27], dtype=int32)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _sub_op.forward(x, y)


def div(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Divides two tensors element-wise.

    This function performs element-wise (true) division on two tensors. It
    supports broadcasting, allowing tensors of different shapes to be combined
    as long as their shapes are compatible. This function also provides the
    implementation of the `/` operator for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar (the dividend).
    y : Tensor | float | int
        The second input tensor or scalar (the divisor). Must be broadcastable
        to the same shape as `x`.

    Returns
    -------
    Tensor
        An tensor containing the result of the element-wise division. The
        result is typically a floating-point tensor.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([10, 20, 30])
    >>> y = nb.tensor([2, 5, 10])
    >>> nb.div(x, y)
    Tensor([5., 4., 3.], dtype=float32)

    >>> x / y
    Tensor([5., 4., 3.], dtype=float32)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _div_op.forward(x, y)


def floordiv(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Performs element-wise floor division on two tensors.

    Floor division is equivalent to `floor(x / y)`, rounding the result
    towards negative infinity. This matches the behavior of Python's `//`
    operator, which this function implements for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar (the dividend).
    y : Tensor | float | int
        The second input tensor or scalar (the divisor). Must be broadcastable
        to the same shape as `x`.

    Returns
    -------
    Tensor
        An tensor containing the result of the element-wise floor division.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([10, -10, 9])
    >>> y = nb.tensor([3, 3, 3])
    >>> nb.floordiv(x, y)
    Tensor([ 3, -4,  3], dtype=int32)

    >>> x // y
    Tensor([ 3, -4,  3], dtype=int32)
    """
    from ..ops.unary import floor

    x = _ensure_tensor(x)
    y = _ensure_tensor(y)

    # Perform regular division then floor
    result = div(x, y)
    return floor(result)


# noqa: A001 - Intentionally shadowing built-in 'pow' for API consistency
def pow(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Computes `x` raised to the power of `y` element-wise.

    This function calculates `x ** y` for each element in the input tensors.
    It supports broadcasting and provides the implementation of the `**`
    operator for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The base tensor or scalar.
    y : Tensor | float | int
        The exponent tensor or scalar. Must be broadcastable to the same shape
        as `x`.

    Returns
    -------
    Tensor
        An tensor containing the result of the element-wise power operation.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([2, 3, 2])
    >>> nb.pow(x, y)
    Tensor([1, 8, 9], dtype=int32)

    >>> x ** y
    Tensor([1, 8, 9], dtype=int32)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _power_op.forward(x, y)


def greater_equal(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Performs element-wise comparison `x >= y`.

    This function compares two tensors element-wise, returning a boolean tensor
    indicating where elements of `x` are greater than or equal to elements
    of `y`. It supports broadcasting and provides the implementation of the
    `>=` operator for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar.
    y : Tensor | float | int
        The second input tensor or scalar. Must be broadcastable to the same
        shape as `x`.

    Returns
    -------
    Tensor
        A boolean tensor containing the result of the element-wise comparison.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1, 5, 3])
    >>> y = nb.tensor([2, 5, 1])
    >>> nb.greater_equal(x, y)
    Tensor([False,  True,  True], dtype=bool)

    >>> x >= y
    Tensor([False,  True,  True], dtype=bool)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _greater_equal_op.forward(x, y)


def maximum(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Computes the element-wise maximum of two tensors.

    This function compares two tensors element-wise and returns a new tensor
    containing the larger of the two elements at each position. It supports
    broadcasting.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar.
    y : Tensor | float | int
        The second input tensor or scalar. Must be broadcastable to the same
        shape as `x`.

    Returns
    -------
    Tensor
        An tensor containing the element-wise maximum of `x` and `y`.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1, 5, 2])
    >>> y = nb.tensor([2, 3, 6])
    >>> nb.maximum(x, y)
    Tensor([2, 5, 6], dtype=int32)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _maximum_op.forward(x, y)


def minimum(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Computes the element-wise minimum of two tensors.

    This function compares two tensors element-wise and returns a new tensor
    containing the smaller of the two elements at each position. It supports
    broadcasting.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar.
    y : Tensor | float | int
        The second input tensor or scalar. Must be broadcastable to the same
        shape as `x`.

    Returns
    -------
    Tensor
        An tensor containing the element-wise minimum of `x` and `y`.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1, 5, 2])
    >>> y = nb.tensor([2, 3, 6])
    >>> nb.minimum(x, y)
    Tensor([1, 3, 2], dtype=int32)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _minimum_op.forward(x, y)


def equal(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Performs element-wise comparison `x == y`.

    This function compares two tensors element-wise, returning a boolean tensor
    indicating where elements of `x` are equal to elements of `y`. It
    supports broadcasting and provides the implementation of the `==` operator
    for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar.
    y : Tensor | float | int
        The second input tensor or scalar. Must be broadcastable to the same
        shape as `x`.

    Returns
    -------
    Tensor
        A boolean tensor containing the result of the element-wise equality
        comparison.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([1, 5, 3])
    >>> nb.equal(x, y)
    Tensor([ True, False,  True], dtype=bool)

    >>> x == y
    Tensor([ True, False,  True], dtype=bool)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _equal_op.forward(x, y)


def not_equal(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Performs element-wise comparison `x != y`.

    This function compares two tensors element-wise, returning a boolean tensor
    indicating where elements of `x` are not equal to elements of `y`. It
    supports broadcasting and provides the implementation of the `!=` operator
    for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The first input tensor or scalar.
    y : Tensor | float | int
        The second input tensor or scalar. Must be broadcastable to the same
        shape as `x`.

    Returns
    -------
    Tensor
        A boolean tensor containing the result of the element-wise inequality
        comparison.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([1, 2, 3])
    >>> y = nb.tensor([1, 5, 3])
    >>> nb.not_equal(x, y)
    Tensor([False,  True, False], dtype=bool)

    >>> x != y
    Tensor([False,  True, False], dtype=bool)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _not_equal_op.forward(x, y)


def mod(x: Tensor | float | int, y: Tensor | float | int) -> Tensor:
    """Computes the element-wise remainder of division.

    This function calculates the remainder of `x / y` element-wise. The
    sign of the result follows the sign of the divisor `y`. It provides the
    implementation of the `%` operator for Nabla tensors.

    Parameters
    ----------
    x : Tensor | float | int
        The dividend tensor or scalar.
    y : Tensor | float | int
        The divisor tensor or scalar. Must be broadcastable to the same shape
        as `x`.

    Returns
    -------
    Tensor
        An tensor containing the element-wise remainder.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.tensor([10, -10, 9])
    >>> y = nb.tensor([3, 3, -3])
    >>> nb.mod(x, y)
    Tensor([ 1,  2, -0], dtype=int32)

    >>> x % y
    Tensor([ 1,  2, -0], dtype=int32)
    """
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    return _mod_op.forward(x, y)
