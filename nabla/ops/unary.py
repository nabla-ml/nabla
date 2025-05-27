"""Unary operations for the Nabla framework."""

from typing import List, Dict, Any
import numpy as np
from max.graph import ops, Value
from max.driver import Tensor
from max.dtype import DType

from ..core.array import Array
from .operation import UnaryOperation


class NegateOp(UnaryOperation):
    """Element-wise negation operation."""

    def __init__(self):
        super().__init__("negate")

    def maxpr(self, args: List[Value], output: Array) -> None:
        output.tensor_value = ops.negate(args[0])

    def eagerxpr(self, args: List[Array], output: Array) -> None:
        np_result = -args[0].get_numpy()
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: List[Array], cotangent: Array, output: Array
    ) -> List[Array]:
        return [negate(cotangent)]

    def jvp_rule(
        self, primals: List[Array], tangents: List[Array], output: Array
    ) -> Array:
        return negate(tangents[0])


def negate(arg: Array) -> Array:
    """Element-wise negation."""
    return _negate_op.forward(arg)


class CastOp(UnaryOperation):
    """Type casting operation."""

    def __init__(self, dtype: DType):
        super().__init__("cast")
        self.target_dtype = dtype

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return input_shape

    def compute_output_dtype(self, arg: Array) -> DType:
        return self.target_dtype

    def forward(self, arg: Array) -> Array:
        """Override forward to set dtype."""
        return super().forward(arg)

    def maxpr(self, args: List[Value], output: Array) -> None:
        output.tensor_value = ops.cast(args[0], output.dtype)

    def eagerxpr(self, args: List[Array], output: Array) -> None:
        np_result = args[0].get_numpy().astype(DType.to_numpy(output.dtype))
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: List[Array], cotangent: Array, output: Array
    ) -> List[Array]:
        return [cast(cotangent, primals[0].dtype)]

    def jvp_rule(
        self, primals: List[Array], tangents: List[Array], output: Array
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

    def maxpr(self, args: List[Value], output: Array) -> None:
        output.tensor_value = ops.sin(args[0])

    def eagerxpr(self, args: List[Array], output: Array) -> None:
        np_result = np.sin(args[0].get_numpy())
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: List[Array], cotangent: Array, output: Array
    ) -> List[Array]:
        # Import here to avoid circular imports
        from .binary import mul

        return [mul(cotangent, cos(primals[0]))]

    def jvp_rule(
        self, primals: List[Array], tangents: List[Array], output: Array
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

    def maxpr(self, args: List[Value], output: Array) -> None:
        output.tensor_value = ops.cos(args[0])

    def eagerxpr(self, args: List[Array], output: Array) -> None:
        np_result = np.cos(args[0].get_numpy())
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: List[Array], cotangent: Array, output: Array
    ) -> List[Array]:
        # Import here to avoid circular imports
        from .binary import mul

        return [negate(mul(cotangent, sin(primals[0])))]

    def jvp_rule(
        self, primals: List[Array], tangents: List[Array], output: Array
    ) -> Array:
        # Import here to avoid circular imports
        from .binary import mul

        return negate(mul(tangents[0], sin(primals[0])))


def cos(arg: Array) -> Array:
    """Element-wise cosine."""
    return _cos_op.forward(arg)


# Add global instances
_negate_op = NegateOp()
_sin_op = SinOp()
_cos_op = CosOp()
