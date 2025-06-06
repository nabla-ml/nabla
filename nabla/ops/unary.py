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
from max.driver import Tensor
from max.dtype import DType
from max.graph import Value, ops

from ..core.array import Array
from .operation import UnaryOperation

# Public API
__all__ = [
    "negate",
    "cast",
    "sin",
    "cos",
    "incr_batch_dim_ctr",
    "decr_batch_dim_ctr",
    "relu",
    "log",
    "exp",
]


class NegateOp(UnaryOperation):
    """Element-wise negation operation."""

    def __init__(self):
        super().__init__("neg")

    def maxpr(self, args: list[Value], output: Array) -> None:
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

    def maxpr(self, args: list[Value], output: Array) -> None:
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

    def maxpr(self, args: list[Value], output: Array) -> None:
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

    def maxpr(self, args: list[Value], output: Array) -> None:
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


class IncrBatchDimCtr(UnaryOperation):
    """Increment batch dimension counter for debugging."""

    def __init__(self):
        super().__init__("incr_batch_dim_ctr")

    def maxpr(self, args: list[Value], output: Array) -> None:
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
    res = IncrBatchDimCtr().forward(arg)
    if len(res.shape) > 0:
        res.batch_dims = res.batch_dims + (res.shape[0],)
        res.shape = res.shape[1:]
    else:
        raise ValueError("Shape is empty, cannot increment batch dimension counter.")
    return res


class DecrBatchDimCtr(UnaryOperation):
    """Decrement batch dimension counter for debugging."""

    def __init__(self):
        super().__init__("decr_batch_dim_ctr")

    def maxpr(self, args: list[Value], output: Array) -> None:
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
    res = DecrBatchDimCtr().forward(arg)
    if res.batch_dims:
        res.shape = (res.batch_dims[-1],) + res.shape
        res.batch_dims = res.batch_dims[:-1]
    else:
        raise ValueError(f"Batch dimensions are empty, cannot decrement. arg_name: {arg.name}")
    return res


class ReLUOp(UnaryOperation):
    """Element-wise ReLU operation."""

    def __init__(self):
        super().__init__("relu")

    def maxpr(self, args: list[Value], output: Array) -> None:
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

    def maxpr(self, args: list[Value], output: Array) -> None:
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

    def maxpr(self, args: list[Value], output: Array) -> None:
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


# Add global instances
_negate_op = NegateOp()
_sin_op = SinOp()
_cos_op = CosOp()
_log_op = LogOp()
_exp_op = ExpOp()
_relu_op = ReLUOp()
