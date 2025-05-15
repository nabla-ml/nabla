# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


from collections import Dict

from nabla.core.device_array import DeviceArray, ArrayImpl, zeros
from nabla.ops.utils import (
    register_binary_op,
    register_any_op,
    get_broadcastedshape,
    RuntimeInfo,
)
from nabla.api.utils import none
from nabla.ops.unary_ops import negate, log
from nabla.ops.view_ops import transpose
from nabla.compiler.graph import Symbol
from nabla.compiler.graph import ops


struct Add:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.add(args[0], args[1])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Add"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(tangent, tangent)

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return add(tangents[0], tangents[1])


fn add(arg0: DeviceArray, arg1: DeviceArray) raises -> DeviceArray:
    return register_binary_op[Add.maxpr, Add.vjp, Add.jvp, Add.eagerxpr](
        arg0, arg1, "add"
    )


struct Mul:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        var _arg0 = args[0]
        var _arg1 = args[1]

        # !!! OPTIONAL: Handle NaN values, has teh following effect: NaN * 0 = 0, MAX usual behavior: NaN * 0 = NaN
        # mask = ops.is_nan(_arg0)
        # zeros = _arg1 - _arg1
        # mask = ops.equal(mask, zeros)
        # _arg0 = ops.select(mask, _arg0, zeros)
        # mask = ops.is_nan(_arg1)
        # zeros = _arg0 - _arg0
        # mask = ops.equal(mask, zeros)
        # _arg1 = ops.select(mask, _arg1, zeros)

        return ops.mul(_arg0, _arg1)

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Mul"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(mul(primals[1], tangent), mul(primals[0], tangent))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return add(mul(primals[1], tangents[0]), mul(primals[0], tangents[1]))


fn mul(arg0: DeviceArray, arg1: DeviceArray) raises -> DeviceArray:
    return register_binary_op[Mul.maxpr, Mul.vjp, Mul.jvp, Mul.eagerxpr](
        arg0, arg1, "mul"
    )


struct Sub:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.sub(args[0], args[1])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Sub"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(tangent, negate(tangent))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return sub(tangents[0], tangents[1])


fn sub(arg0: DeviceArray, arg1: DeviceArray) raises -> DeviceArray:
    return register_binary_op[Sub.maxpr, Sub.vjp, Sub.jvp, Sub.eagerxpr](
        arg0, arg1, "sub"
    )


struct Div:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.div(args[0], args[1])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Div"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(
            div(tangent, primals[1]),
            div(negate(mul(tangent, primals[0])), mul(primals[1], primals[1])),
        )

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return sub(
            div(tangents[0], primals[1]),
            div(mul(primals[0], tangents[1]), primals[1] ** Float32(2.0)),
        )


fn div(arg0: DeviceArray, arg1: DeviceArray) raises -> DeviceArray:
    return register_binary_op[Div.maxpr, Div.vjp, Div.jvp, Div.eagerxpr](
        arg0, arg1, "div"
    )


struct Matmul:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.matmul(args[0], args[1])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Matmul"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var lhs = matmul(tangent, transpose(primals[1], -1, -2))
        var rhs = matmul(transpose(primals[0], -1, -2), tangent)
        return List(lhs, rhs)

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var lhs = matmul(tangents[0], primals[1])
        var rhs = matmul(primals[0], tangents[1])
        return add(lhs, rhs)


fn matmul(_arg0: DeviceArray, _arg1: DeviceArray) raises -> DeviceArray:
    var arg0 = _arg0
    var arg1 = _arg1

    var arg0_offset = arg0.batch_dim_ctr() if arg0.batch_dim_ctr() != none else 0
    var arg1_offset = arg1.batch_dim_ctr() if arg1.batch_dim_ctr() != none else 0

    var arg0_batch_dims = arg0.shape()[:arg0_offset]
    var arg1_batch_dims = arg1.shape()[:arg1_offset]
    var arg0_true_dims = arg0.shape()[arg0_offset:]
    var arg1_true_dims = arg1.shape()[arg1_offset:]

    var res_batch_dim = get_broadcastedshape(
        arg0_batch_dims,
        arg1_batch_dims,
    )
    var res_true_dim = get_broadcastedshape(
        arg0_true_dims,
        arg1_true_dims,
        right_offset=2,
    )
    res_true_dim[-2] = arg0_true_dims[-2]
    res_true_dim[-1] = arg1_true_dims[-1]
    var new_shape = res_batch_dim + res_true_dim

    return register_any_op[
        Matmul.maxpr, Matmul.vjp, Matmul.jvp, Matmul.eagerxpr
    ](
        List(arg0, arg1),
        "matmul",
        new_shape,
    )


struct GreaterThan:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.greater(args[0], args[1])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for GreaterThan"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        raise "VJP for GreaterThan is not implemented"

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        raise "JVP for GreaterThan is not implemented"


fn gt(arg0: DeviceArray, arg1: DeviceArray) raises -> DeviceArray:
    return register_binary_op[
        GreaterThan.maxpr,
        GreaterThan.vjp,
        GreaterThan.jvp,
        GreaterThan.eagerxpr,
    ](arg0, arg1, "gt")


struct Pow:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.pow(args[0], args[1])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Pow"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var base = primals[0]
        var exp = primals[1]
        var expdtype = exp.dtype()
        return List(
            mul(tangent, mul(exp, pow(base, exp - 1))),
            zeros(
                shape=List(1),
                dtype=expdtype,
                requires_pullback=False,
            ),
        )

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var x = primals[0]
        var y = primals[1]
        return tangents[0] * y * pow(x, y - 1) + tangents[1] * pow(x, y) * log(
            x
        )


fn pow(arg0: DeviceArray, arg1: DeviceArray) raises -> DeviceArray:
    return register_binary_op[Pow.maxpr, Pow.vjp, Pow.jvp, Pow.eagerxpr](
        arg0, arg1, "pow"
    )
