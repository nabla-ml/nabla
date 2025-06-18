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
from max.driver import Device, Tensor
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
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [negate(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return negate(tangents[0])


def negate(arg: Array) -> Array:
    """Element-wise negation."""
    return _negate_op.forward(arg)


class CastOp(UnaryOperation):
    """Type casting operation."""

    def __init__(self, dtype: DType):
        super().__init__(f"convert_element_type[new_dtype={dtype}]")
        self.target_dtype = dtype

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Compatible signature - output shape same as input shape."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Cast operation requires 1 input shape, got {len(input_shapes)}"
            )
        return input_shapes[0]

    def compute_output_dtype(self, arg: Array) -> DType:
        return self.target_dtype

    def forward(self, *args: Array) -> Array:
        """Override forward to set dtype with compatible signature."""
        if len(args) != 1:
            raise ValueError(f"Cast operation requires 1 argument, got {len(args)}")
        return super().forward(*args)

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.cast(args[0], output.dtype)

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = args[0].to_numpy().astype(DType.to_numpy(output.dtype))
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [cast(cotangent, primals[0].dtype)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return cast(tangents[0], output.dtype)


def cast(arg: Array, dtype: DType) -> Array:
    """Cast array to different dtype."""
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
        output.impl = Tensor.from_numpy(np_result)

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
    """Element-wise sine."""
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
        output.impl = Tensor.from_numpy(np_result)

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
    """Element-wise cosine."""
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
        output.impl = Tensor.from_numpy(np_result)

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
    """Element-wise hyperbolic tangent."""
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
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import greater_equal, mul
        from .creation import ones_like, zeros

        # d/dx |x| = sign(x) = 1 if x > 0, -1 if x < 0, undefined at x = 0
        # We use the convention that sign(0) = 0
        zero = zeros((), dtype=primals[0].dtype)
        pos_mask = greater_equal(primals[0], zero)
        pos_mask_casted = cast(pos_mask, cotangent.dtype)

        # Create sign: 1 for positive, -1 for negative
        ones_tensor = ones_like(primals[0])
        sign = 2.0 * pos_mask_casted - ones_tensor

        return [mul(cotangent, sign)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import greater_equal, mul
        from .creation import ones_like, zeros

        # d/dx |x| = sign(x)
        zero = zeros((), dtype=primals[0].dtype)
        pos_mask = greater_equal(primals[0], zero)
        pos_mask_casted = cast(pos_mask, tangents[0].dtype)

        # Create sign: 1 for positive, -1 for negative
        ones_tensor = ones_like(primals[0])
        sign = 2.0 * pos_mask_casted - ones_tensor

        return mul(tangents[0], sign)


def abs(arg: Array) -> Array:
    """Element-wise absolute value."""
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
        output.impl = Tensor.from_numpy(np_result)

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
    """Element-wise floor function."""
    return _floor_op.forward(arg)


class LogicalNotOp(UnaryOperation):
    """Element-wise logical NOT operation for boolean arrays."""

    def __init__(self):
        super().__init__("logical_not")

    def compute_output_dtype(self, arg: Array) -> DType:
        """Logical NOT always returns boolean dtype."""
        return DType.bool

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        # Use MAX's logical not operation
        output.tensor_value = ops.logical_not(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.logical_not(args[0].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

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
    """Element-wise logical NOT operation."""
    return _logical_not_op.forward(arg)


class SigmoidOp(UnaryOperation):
    """Element-wise sigmoid operation."""

    def __init__(self):
        super().__init__("sigmoid")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        # Sigmoid = 1 / (1 + exp(-x))
        # Use MAX's built-in sigmoid if available, otherwise construct from primitives
        output.tensor_value = ops.sigmoid(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        # Numerically stable sigmoid implementation
        x = args[0].to_numpy()
        # For positive values: 1 / (1 + exp(-x))
        # For negative values: exp(x) / (1 + exp(x))
        np_result = np.where(
            x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x))
        )
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import mul, sub
        from .creation import ones_like

        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = output * (1 - output)
        ones_like_output = ones_like(output)
        one_minus_output = sub(ones_like_output, output)
        derivative = mul(output, one_minus_output)
        return [mul(cotangent, derivative)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import mul, sub
        from .creation import ones_like

        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        ones_like_output = ones_like(output)
        one_minus_output = sub(ones_like_output, output)
        derivative = mul(output, one_minus_output)
        return mul(tangents[0], derivative)


def sigmoid(arg: Array) -> Array:
    """Element-wise sigmoid function."""
    return _sigmoid_op.forward(arg)


class IncrBatchDimCtr(UnaryOperation):
    """Increment batch dimension counter for debugging."""

    def __init__(self, arg_batch_dims: tuple[int, ...], arg_shape: tuple[int, ...]):
        super().__init__("incr_batch_dim_ctr")
        self.arg_batch_dims = arg_batch_dims
        self.arg_shape = arg_shape

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Output shape is the same as input shape."""
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
        output.impl = args[0].impl

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [decr_batch_dim_ctr(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return incr_batch_dim_ctr(tangents[0])


def incr_batch_dim_ctr(arg: Array) -> Array:
    """Increment batch dimension counter for debugging."""
    return IncrBatchDimCtr(arg.batch_dims, arg.shape).forward(arg)


class DecrBatchDimCtr(UnaryOperation):
    """Decrement batch dimension counter for debugging."""

    def __init__(self, arg_batch_dims: tuple[int, ...], arg_shape: tuple[int, ...]):
        super().__init__("decr_batch_dim_ctr")
        self.arg_batch_dims = arg_batch_dims
        self.arg_shape = arg_shape

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        """Output shape is the same as input shape."""
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
        output.impl = args[0].impl

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [incr_batch_dim_ctr(cotangent)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return decr_batch_dim_ctr(tangents[0])


def decr_batch_dim_ctr(arg: Array) -> Array:
    """Decrement batch dimension counter for debugging."""
    return DecrBatchDimCtr(arg.batch_dims, arg.shape).forward(arg)


class ReLUOp(UnaryOperation):
    """Element-wise ReLU operation."""

    def __init__(self):
        super().__init__("relu")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.relu(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.maximum(0, args[0].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import greater_equal, mul
        from .creation import zeros

        # Create zero with same dtype as primal to avoid dtype mismatch
        zero = zeros((), dtype=primals[0].dtype)
        mask = greater_equal(primals[0], zero)
        mask_casted = cast(mask, cotangent.dtype)
        return [mul(cotangent, mask_casted)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        # Import here to avoid circular imports
        from .binary import greater_equal, mul
        from .creation import zeros

        # ReLU derivative is 1 for x > 0, 0 for x <= 0
        zero = zeros((), dtype=primals[0].dtype)
        mask = greater_equal(primals[0], zero)
        mask_casted = cast(mask, tangents[0].dtype)
        return mul(tangents[0], mask_casted)


def relu(arg: Array) -> Array:
    """Element-wise ReLU (Rectified Linear Unit) function."""
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
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

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
    """Element-wise natural logarithm."""
    return _log_op.forward(arg)


class ExpOp(UnaryOperation):
    """Element-wise exponential operation."""

    def __init__(self):
        super().__init__("exp")

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.exp(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.exp(args[0].to_numpy())
        # Ensure result is an array, not a scalar
        if np.isscalar(np_result):
            np_result = np.array(np_result)
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        from .binary import mul

        # d/dx exp(x) = exp(x), and output = exp(x)
        return [mul(cotangent, output)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        from .binary import mul

        # d/dx exp(x) = exp(x)
        return mul(output, tangents[0])


def exp(arg: Array) -> Array:
    """Element-wise exponential function."""
    return _exp_op.forward(arg)


def sqrt(arg: Array) -> Array:
    """Element-wise square root function.

    Implemented as pow(arg, 0.5) for compatibility with the automatic
    differentiation system.
    """
    from .binary import pow as binary_pow
    from .creation import array

    # Create 0.5 as a scalar Array
    half = array([0.5], dtype=arg.dtype)
    return binary_pow(arg, half)


# class ZerosLikeOp(UnaryOperation):
#     """Create an array of zeros with the same shape and dtype as the input."""

#     def __init__(self):
#         super().__init__("zeros_like")

#     def compute_output_shape(self, *input_shapes: tuple) -> tuple:
#         """Output shape is the same as input shape."""
#         if len(input_shapes) != 1:
#             raise ValueError(f"ZerosLikeOp requires 1 input shape, got {len(input_shapes)}")
#         return input_shapes[0]

#     def compute_output_dtype(self, arg: Array) -> DType:
#         """Output dtype is the same as input dtype."""
#         return arg.dtype

#     def maxpr(self, args: list[TensorValue], output: Array) -> None:
#         output.tensor_value = ops.shape_to_tensor(args[0].shape)

#     def eagerxpr(self, args: list[Array], output: Array) -> None:
#         np_result = np.zeros(args[0].shape, dtype=args[0].dtype.to_numpy())
#         output.impl = Tensor.from_numpy(np_result)

#     def vjp_rule(
#         self, primals: list[Array], cotangent: Array, output: Array
#     ) -> list[Array]:
#         return [zeros_like(cotangent)]

#     def jvp_rule(
#         self, primals: list[Array], tangents: list[Array], output: Array
#     ) -> Array:
#         return zeros_like(tangents[0])


class TransferToOp(UnaryOperation):
    """Transfer operation to a different device."""

    def __init__(self, arg_device: Device, target_device: Device):
        super().__init__(f"transfer_to[{target_device}]")
        self.arg_device = arg_device
        self.target_device = target_device

    def forward(self, *args: Array) -> Array:
        """Forward pass for unary operations."""
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

        return res

    def maxpr(self, args: list[TensorValue], output: Array) -> None:
        output.tensor_value = ops.transfer_to(
            args[0], DeviceRef.from_device(self.target_device)
        )

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        output.impl = args[0].impl

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [transfer_to(cotangent, self.arg_device)]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return transfer_to(tangents[0], self.target_device)


def transfer_to(arg: Array, device: Device) -> Array:
    """Transfer an array to a different device."""
    if not isinstance(device, Device):
        raise TypeError(f"Device must be an instance of Device, got {type(device)}")
    # if arg.device.id == device.id:
    #     return arg
    return TransferToOp(arg.device, device).forward(arg)


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
