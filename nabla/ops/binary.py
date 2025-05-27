# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import numpy as np
from max.driver import Tensor
from max.graph import Value, ops

from ..core.array import Array
from .operation import BinaryOperation


def _ensure_array(value) -> Array:
    """Convert scalar values to Arrays."""
    if isinstance(value, Array):
        return value
    elif isinstance(value, int | float):
        from .creation import array

        return array([value])
    else:
        raise TypeError(f"Cannot convert {type(value)} to Array")


class AddOp(BinaryOperation):
    """Addition operation."""

    def __init__(self):
        super().__init__("add")

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.add(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.add(args[0].get_numpy(), args[1].get_numpy())
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

    def maxpr(self, args: list[Value], output: Array) -> None:
        output.tensor_value = ops.mul(args[0], args[1])

    def eagerxpr(self, args: list[Array], output: Array) -> None:
        np_result = np.multiply(args[0].get_numpy(), args[1].get_numpy())
        output.impl = Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        return [mul(cotangent, primals[1]), mul(cotangent, primals[0])]

    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        return add(mul(primals[0], tangents[1]), mul(primals[1], tangents[0]))


# Global operation instances
_add_op = AddOp()
_mul_op = MulOp()


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
