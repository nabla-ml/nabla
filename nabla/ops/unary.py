# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
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


class NegateOp(UnaryOperation):
    """Element-wise negation operation."""

    def __init__(self):
        super().__init__("neg")

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.negate(args[0])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = -args[0].get_numpy()
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
        np_result = args[0].get_numpy().astype(DType.to_numpy(output.dtype))
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
        np_result = np.sin(args[0].get_numpy())
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        # Import here to avoid circular imports
        from .binary import mul

        return [mul(cotangent, cos(primals[0]))]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        # Import here to avoid circular imports
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
        np_result = np.cos(args[0].get_numpy())
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        # Import here to avoid circular imports
        from .binary import mul

        return [negate(mul(cotangent, sin(primals[0])))]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        # Import here to avoid circular imports
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
        output.tensor_value = args[0].tensor_value  # No operation, just pass through

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
    res.batch_dim_ctr += 1
    return res


class DecrBatchDimCtr(UnaryOperation):
    """Decrement batch dimension counter for debugging."""

    def __init__(self):
        super().__init__("decr_batch_dim_ctr")

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = args[0].tensor_value  # No operation, just pass through

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
    if arg.batch_dim_ctr <= 0:
        raise ValueError("Batch dimension counter cannot be decremented below zero.")
    res = DecrBatchDimCtr().forward(arg)
    res.batch_dim_ctr -= 1
    return res


# Add global instances
_negate_op = NegateOp()
_sin_op = SinOp()
_cos_op = CosOp()
