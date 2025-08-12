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

"""Unary operations for the Nabla framework."""

import numpy as np
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from ..core.array import Array
from .operation import UnaryOperation

# Public API
__all__ = [
    "negate",
    "cast",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "abs",
    "floor",
    "logical_not",
    "incr_batch_dim_ctr",
    "decr_batch_dim_ctr",
    "relu",
    "log",
    "exp",
    "sqrt",
    "transfer_to",
]


class NegateOp(UnaryOperation):
    """Element-wise negation operation."""

    def __init__(self):
        super().__init__("neg")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.negate(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = -args[0].to_numpy()
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [negate(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return negate(tangents[0])


def negate(arg: Array) -> Array:
    """Computes the element-wise numerical negative of an array.

    This function returns a new array with each element being the negation
    of the corresponding element in the input array. It also provides the
    implementation for the unary `-` operator on Nabla arrays.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        An array containing the negated elements.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([1, -2, 3.5])
    >>> nb.negate(x)
    Array([-1.,  2., -3.5], dtype=float32)

    Using the `-` operator:
    >>> -x
    Array([-1.,  2., -3.5], dtype=float32)
    """
    return _negate_op.forward(arg)


class CastOp(UnaryOperation):
    """Type casting operation."""

    def __init__(self, dtype: DType):
        super().__init__(f"convert_element_type[new_dtype={dtype}]")
        self.target_dtype = dtype

    def compute_output_dtype(self, arg: Array) -> DType:
        return self.target_dtype

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.cast(args[0], output.dtype)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = args[0].to_numpy().astype(DType.to_numpy(output.dtype))
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [cast(cotangent, primals[0].dtype)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return cast(tangents[0], output.dtype)


def cast(arg: Array, dtype: DType) -> Array:
    """Casts an array to a specified data type.

    This function creates a new array with the same shape as the input but
    with the specified data type (`dtype`).

    Parameters
    ----------
    arg : Array
        The input array to be cast.
    dtype : DType
        The target Nabla data type (e.g., `nb.float32`, `nb.int32`).

    Returns
    -------
    Array
        A new array with the elements cast to the specified `dtype`.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([1, 2, 3])
    >>> x.dtype
    int32
    >>> y = nb.cast(x, nb.float32)
    >>> y
    Array([1., 2., 3.], dtype=float32)
    """
    if not isinstance(dtype, DType):
        raise TypeError(f"Dtype must be an instance of DType, got {type(dtype)}")

    op = CastOp(dtype)
    return op.forward(arg)


class SinOp(UnaryOperation):
    """Element-wise sine operation."""

    def __init__(self):
        super().__init__("sin")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.sin(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.sin(args[0].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import mul

        return [mul(cotangent, cos(primals[0]))]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import mul

        return mul(tangents[0], cos(primals[0]))


def sin(arg: Array, dtype: DType | None = None) -> Array:
    """Computes the element-wise sine of an array.

    Parameters
    ----------
    arg : Array
        The input array. Input is expected to be in radians.
    dtype : DType | None, optional
        If provided, the output array will be cast to this data type.

    Returns
    -------
    Array
        An array containing the sine of each element in the input.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([0, 1.5707963, 3.1415926])
    >>> nb.sin(x)
    Array([0.0000000e+00, 1.0000000e+00, -8.7422780e-08], dtype=float32)
    """
    res = _sin_op.forward(arg)
    if dtype:
        return cast(res, dtype)
    return res


class CosOp(UnaryOperation):
    """Element-wise cosine operation."""

    def __init__(self):
        super().__init__("cos")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.cos(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.cos(args[0].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import mul

        return [negate(mul(cotangent, sin(primals[0])))]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import mul

        return negate(mul(tangents[0], sin(primals[0])))


def cos(arg: Array) -> Array:
    """Computes the element-wise cosine of an array.

    Parameters
    ----------
    arg : Array
        The input array. Input is expected to be in radians.

    Returns
    -------
    Array
        An array containing the cosine of each element in the input.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([0, 1.5707963, 3.1415926])
    >>> nb.cos(x)
    Array([ 1.000000e+00, -4.371139e-08, -1.000000e+00], dtype=float32)
    """
    return _cos_op.forward(arg)


class TanhOp(UnaryOperation):
    """Element-wise hyperbolic tangent operation."""

    def __init__(self):
        super().__init__("tanh")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.tanh(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.tanh(args[0].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import mul, sub
        from .creation import ones_like

        # d/dx tanh(x) = 1 - tanh²(x) = 1 - output²
        ones_like_output = ones_like(output)
        tanh_squared = mul(output, output)
        derivative = sub(ones_like_output, tanh_squared)
        return [mul(cotangent, derivative)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import mul, sub
        from .creation import ones_like

        # d/dx tanh(x) = 1 - tanh²(x)
        ones_like_output = ones_like(output)
        tanh_squared = mul(output, output)
        derivative = sub(ones_like_output, tanh_squared)
        return mul(tangents[0], derivative)


def tanh(arg: Array) -> Array:
    """Computes the element-wise hyperbolic tangent of an array.

    The tanh function is a common activation function in neural networks,
    squashing values to the range `[-1, 1]`.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        An array containing the hyperbolic tangent of each element.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([-1.0, 0.0, 1.0, 20.0])
    >>> nb.tanh(x)
    Array([-0.7615942,  0.       ,  0.7615942,  1.       ], dtype=float32)
    """
    return _tanh_op.forward(arg)


class AbsOp(UnaryOperation):
    """Element-wise absolute value operation."""

    def __init__(self):
        super().__init__("abs")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.abs(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.abs(args[0].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import mul

        x = primals[0]
        abs_x = output
        eps = 1e-12
        abs_x_safe = abs_x + eps
        from .binary import div

        sign = div(x, abs_x_safe)

        return [mul(cotangent, sign)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import div, mul
        x = primals[0]
        abs_x = output
        eps = 1e-12
        abs_x_safe = abs_x + eps
        sign = div(x, abs_x_safe)
        return mul(tangents[0], sign)


def abs(arg: Array) -> Array:
    """Computes the element-wise absolute value of an array.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        An array containing the absolute value of each element.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([-1.5, 0.0, 2.5])
    >>> nb.abs(x)
    Array([1.5, 0. , 2.5], dtype=float32)
    """
    return _abs_op.forward(arg)


class FloorOp(UnaryOperation):
    """Element-wise floor operation."""

    def __init__(self):
        super().__init__("floor")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.floor(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.floor(args[0].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .creation import zeros_like

        return [zeros_like(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .creation import zeros_like

        return zeros_like(tangents[0])


def floor(arg: Array) -> Array:
    """Computes the element-wise floor of an array.

    The floor of a scalar `x` is the largest integer `i` such that `i <= x`.
    This function is not differentiable and its gradient is zero everywhere.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        An array containing the floor of each element.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([-1.7, -0.2, 0.2, 1.7])
    >>> nb.floor(x)
    Array([-2., -1.,  0.,  1.], dtype=float32)
    """
    return _floor_op.forward(arg)


class LogicalNotOp(UnaryOperation):
    """Element-wise logical NOT operation for boolean arrays."""

    def __init__(self):
        super().__init__("logical_not")

    def compute_output_dtype(self, arg: Array) -> DType:
        return DType.bool

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        input_tensor = args[0]
        if input_tensor.dtype != DType.bool:
            input_tensor = ops.cast(input_tensor, DType.bool)
        output.tensor_value = ops.logical_not(input_tensor)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        import numpy as np

        np_result = np.logical_not(args[0].to_numpy())
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        if np_result.shape == () and np_result.dtype == bool:
            np_result_1d = np.array([np_result.item()], dtype=bool)
            output.impl_(np_result_1d)
            output.shape = ()
        else:
            output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .creation import zeros_like

        return [zeros_like(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .creation import zeros_like

        return zeros_like(tangents[0])


def logical_not(arg: Array) -> Array:
    """Computes the element-wise logical NOT of a boolean array.

    This function inverts the boolean value of each element in the input array.
    Input arrays of non-boolean types will be cast to boolean first.

    Parameters
    ----------
    arg : Array
        The input boolean array.

    Returns
    -------
    Array
        A boolean array containing the inverted values.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([True, False, True])
    >>> nb.logical_not(x)
    Array([False,  True, False], dtype=bool)
    """
    return _logical_not_op.forward(arg)


class SigmoidOp(UnaryOperation):
    """Element-wise sigmoid operation."""

    def __init__(self):
        super().__init__("sigmoid")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.sigmoid(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        x = args[0].to_numpy()
        np_result = np.where(
            x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x))
        )
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import mul, sub
        from .creation import ones_like

        ones_like_output = ones_like(output)
        one_minus_output = sub(ones_like_output, output)
        derivative = mul(output, one_minus_output)
        return [mul(cotangent, derivative)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import mul, sub
        from .creation import ones_like

        ones_like_output = ones_like(output)
        one_minus_output = sub(ones_like_output, output)
        derivative = mul(output, one_minus_output)
        return mul(tangents[0], derivative)


def sigmoid(arg: Array) -> Array:
    """Computes the element-wise sigmoid function.

    The sigmoid function, defined as `1 / (1 + exp(-x))`, is a common
    activation function that squashes values to the range `(0, 1)`.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        An array containing the sigmoid of each element.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([-1.0, 0.0, 1.0, 20.0])
    >>> nb.sigmoid(x)
    Array([0.26894143, 0.5       , 0.7310586 , 1.        ], dtype=float32)
    """
    return _sigmoid_op.forward(arg)


class IncrBatchDimCtr(UnaryOperation):
    """Increment batch dimension counter for debugging."""

    def __init__(self, arg_batch_dims: tuple[int, ...], arg_shape: tuple[int, ...]):
        super().__init__("incr_batch_dim_ctr")
        self.arg_batch_dims = arg_batch_dims
        self.arg_shape = arg_shape

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        if not self.arg_shape:
            raise ValueError(
                f"IncrBatchDimCtr requires a non-empty arg_shape, got {self.arg_shape}"
            )
        return self.arg_shape[1:]

    def compute_output_batch_dims(self, *input_batch_dims):
        if not self.arg_shape:
            raise ValueError(
                f"IncrBatchDimCtr requires a non-empty arg_shape, got {self.arg_shape}"
            )
        return self.arg_batch_dims + (self.arg_shape[0],)

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = args[0]

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        output.impl_(args[0]._impl)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [decr_batch_dim_ctr(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return incr_batch_dim_ctr(tangents[0])


def incr_batch_dim_ctr(arg: Array) -> Array:
    """Moves the leading axis from `shape` to `batch_dims`. (Internal use)

    This is an internal-use function primarily for developing function
    transformations like `vmap`. It re-interprets the first dimension of the
    array's logical shape as a new batch dimension.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        A new array with an additional batch dimension.
    """
    return IncrBatchDimCtr(arg.batch_dims, arg.shape).forward(arg)


class DecrBatchDimCtr(UnaryOperation):
    """Decrement batch dimension counter for debugging."""

    def __init__(self, arg_batch_dims: tuple[int, ...], arg_shape: tuple[int, ...]):
        super().__init__("decr_batch_dim_ctr")
        self.arg_batch_dims = arg_batch_dims
        self.arg_shape = arg_shape

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        if not self.arg_batch_dims:
            raise ValueError(
                f"DecrBatchDimCtr requires a non-empty arg_batch_dims, got {self.arg_batch_dims}"
            )
        return (self.arg_batch_dims[-1],) + self.arg_shape

    def compute_output_batch_dims(self, *input_batch_dims):
        if not self.arg_batch_dims:
            raise ValueError(
                f"DecrBatchDimCtr requires a non-empty arg_batch_dims, got {self.arg_batch_dims}"
            )
        return self.arg_batch_dims[:-1]

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = args[0]

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        output.impl_(args[0]._impl)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [incr_batch_dim_ctr(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return decr_batch_dim_ctr(tangents[0])


def decr_batch_dim_ctr(arg: Array) -> Array:
    """Moves the last `batch_dim` to be the leading axis of `shape`. (Internal use)

    This is an internal-use function primarily for developing function
    transformations like `vmap`. It re-interprets the last batch dimension
    as the new first dimension of the array's logical shape.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        A new array with one fewer batch dimension.
    """
    return DecrBatchDimCtr(arg.batch_dims, arg.shape).forward(arg)


class ReLUOp(UnaryOperation):
    """Element-wise ReLU operation."""

    def __init__(self):
        super().__init__("relu")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.relu(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.maximum(0, args[0].to_numpy())
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import div, mul
        x = primals[0]
        eps = 1e-12
        x_abs = abs(x)
        x_safe = x_abs + eps
        derivative = div(output, x_safe)
        return [mul(cotangent, derivative)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import div, mul
        x = primals[0]
        eps = 1e-12
        x_abs = abs(x)
        x_safe = x_abs + eps
        derivative = div(output, x_safe)
        return mul(tangents[0], derivative)


def relu(arg: Array) -> Array:
    """Computes the element-wise Rectified Linear Unit (ReLU) function.

    The ReLU function is defined as `max(0, x)`. It is a widely used
    activation function in neural networks.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        An array containing the result of the ReLU operation.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([-2.0, -0.5, 0.0, 1.0, 2.0])
    >>> nb.relu(x)
    Array([0., 0., 0., 1., 2.], dtype=float32)
    """
    return _relu_op.forward(arg)


class LogOp(UnaryOperation):
    """Element-wise natural logarithm operation."""

    def __init__(self):
        super().__init__("log")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.log(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        input_array = args[0].to_numpy()
        epsilon = 1e-15
        safe_input = np.maximum(input_array, epsilon)
        np_result = np.log(safe_input)
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import div

        return [div(cotangent, primals[0])]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import div

        return div(tangents[0], primals[0])


def log(arg: Array) -> Array:
    """Computes the element-wise natural logarithm (base e).

    This function calculates `log(x)` for each element `x` in the input array.
    For numerical stability with non-positive inputs, a small epsilon is
    added to ensure the input to the logarithm is positive.

    Parameters
    ----------
    arg : Array
        The input array. Values should be positive.

    Returns
    -------
    Array
        An array containing the natural logarithm of each element.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([1.0, 2.71828, 10.0])
    >>> nb.log(x)
    Array([0.       , 0.9999993, 2.3025851], dtype=float32)
    """
    return _log_op.forward(arg)


class ExpOp(UnaryOperation):
    """Element-wise exponential operation."""

    def __init__(self):
        super().__init__("exp")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.exp(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.exp(args[0].to_numpy())
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl_(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import mul

        return [mul(cotangent, output)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import mul
        return mul(output, tangents[0])


def exp(arg: Array) -> Array:
    """Computes the element-wise exponential function (e^x).

    This function calculates the base-e exponential of each element in the
    input array.

    Parameters
    ----------
    arg : Array
        The input array.

    Returns
    -------
    Array
        An array containing the exponential of each element.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([0.0, 1.0, 2.0])
    >>> nb.exp(x)
    Array([1.       , 2.7182817, 7.389056 ], dtype=float32)
    """
    return _exp_op.forward(arg)


def sqrt(arg: Array) -> Array:
    """Computes the element-wise non-negative square root of an array.

    This function is implemented as `nabla.pow(arg, 0.5)` to ensure it is
    compatible with the automatic differentiation system.

    Parameters
    ----------
    arg : Array
        The input array. All elements must be non-negative.

    Returns
    -------
    Array
        An array containing the square root of each element.

    Examples
    --------
    >>> import nabla as nb
    >>> x = nb.array([0.0, 4.0, 9.0])
    >>> nb.sqrt(x)
    Array([0., 2., 3.], dtype=float32)
    """
    from .binary import pow as binary_pow
    from .creation import array

    half = array(0.5, dtype=arg.dtype)
    return binary_pow(arg, half)


class TransferToOp(UnaryOperation):
    """Transfer operation to a different device."""

    def __init__(self, arg_device: Device, target_device: Device):
        super().__init__(f"transfer_to[{target_device}]")
        self.arg_device = arg_device
        self.target_device = target_device

    def forward(self, *args: Array) -> Array:
        if len(args) != 1:
            raise ValueError(f"Unary operation requires 1 argument, got {len(args)}")
        arg = args[0]

        output_shape = self.compute_output_shape(arg.shape)
        output_batch_dims = self.compute_output_batch_dims(arg.batch_dims)
        output_dtype = self.compute_output_dtype(arg)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,
            device=self.target_device,
            materialize=False,
            name=self.name,
            batch_dims=output_batch_dims,
        )

        res.set_maxpr(self.maxpr)
        res.add_arguments(arg)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule
        res.custom_kernel_path = self.custom_kernel_path()

        if not res.stage_realization:
            self.eagerxpr([arg], res)

        res.creator_op = self
        return res

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.transfer_to(
            args[0], DeviceRef.from_device(self.target_device)
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        output.impl_(args[0].impl.to(self.target_device))

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [transfer_to(cotangent, self.arg_device)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return transfer_to(tangents[0], self.target_device)


def transfer_to(arg: Array, device: Device) -> Array:
    """Transfers an array to a different compute device.

    This function moves the data of a Nabla array to the specified device
    (e.g., from CPU to GPU). If the array is already on the target device,
    it is returned unchanged.

    Parameters
    ----------
    arg : Array
        The input array to transfer.
    device : Device
        The target device instance (e.g., `nb.Device.cpu()`, `nb.Device.gpu()`).

    Returns
    -------
    Array
        A new array residing on the target device.
    """
    if not isinstance(device, Device):
        raise TypeError(f"Device must be an instance of Device, got {type(device)}")
    if arg.logical_device == device:
        return arg
    return TransferToOp(arg.logical_device, device).forward(arg)


# Add global instances
_negate_op = NegateOp()
_sin_op = SinOp()
_cos_op = CosOp()
_tanh_op = TanhOp()
_abs_op = AbsOp()
_floor_op = FloorOp()
_logical_not_op = LogicalNotOp()
_sigmoid_op = SigmoidOp()
_log_op = LogOp()
_exp_op = ExpOp()
_relu_op = ReLUOp()