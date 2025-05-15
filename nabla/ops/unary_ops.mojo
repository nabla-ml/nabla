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

from nabla.core.device_array import DeviceArray, ArrayImpl
from nabla.ops.utils import register_unary_op, RuntimeInfo
from nabla.ops.binary_ops import mul, div
from nabla.compiler.graph import Symbol
from nabla.compiler.graph import ops

struct Sin:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.sin(args[0])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Sin"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(mul(cos(primals[0]), tangent))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        result: DeviceArray,
    ) raises -> DeviceArray:
        return mul(cos(primals[0]), tangents[0])


fn sin(arg: DeviceArray) raises -> DeviceArray:
    return register_unary_op[Sin.maxpr, Sin.vjp, Sin.jvp, Sin.eagerxpr](
        arg, "sin"
    )


struct Cast:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.cast(args[0], array.dtype())

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Cast"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(cast(tangent, primals[0].impl[].spec.dtype()))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return cast(tangents[0], array.impl[].spec.dtype())


fn cast(arg: DeviceArray, dtype: DType) raises -> DeviceArray:
    return register_unary_op[Cast.maxpr, Cast.vjp, Cast.jvp, Cast.eagerxpr](
        arg, "cast" + String(dtype)
    )


struct Negate:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return -args[0]

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Negate"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(negate(tangent))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return negate(tangents[0])


fn negate(arg: DeviceArray) raises -> DeviceArray:
    return register_unary_op[
        Negate.maxpr, Negate.vjp, Negate.jvp, Negate.eagerxpr
    ](arg, "negate")


struct Cos:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.cos(args[0])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Cos"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(negate(mul(sin(primals[0]), tangent)))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return negate(mul(sin(primals[0]), tangents[0]))


fn cos(arg: DeviceArray) raises -> DeviceArray:
    return register_unary_op[Cos.maxpr, Cos.vjp, Cos.jvp, Cos.eagerxpr](
        arg, "cos"
    )


struct ReLU:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.relu(args[0])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for ReLU"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var mask = gt_zero(primals[0])
        return List(mul(mask, tangent))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var mask = gt_zero(primals[0])
        return mul(mask, tangents[0])


fn relu(arg: DeviceArray) raises -> DeviceArray:
    return register_unary_op[ReLU.maxpr, ReLU.vjp, ReLU.jvp, ReLU.eagerxpr](
        arg, "relu"
    )


struct Log:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return ops.log(args[0])

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Log"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(div(tangent, primals[0]))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return div(tangents[0], primals[0])


fn log(arg: DeviceArray) raises -> DeviceArray:
    return register_unary_op[Log.maxpr, Log.vjp, Log.jvp, Log.eagerxpr](
        arg, "log"
    )


struct GreaterThanZero:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        var zeros = args[0] - args[0]
        return ops.greater(args[0], zeros)

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for GreaterThanZero"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        raise "VJP for GreaterThanZero is not implemented"

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        raise "JVP for GreaterThanZero is not implemented"


fn gt_zero(arg: DeviceArray) raises -> DeviceArray:
    return register_unary_op[
        GreaterThanZero.maxpr,
        GreaterThanZero.vjp,
        GreaterThanZero.jvp,
        GreaterThanZero.eagerxpr,
    ](arg, "gt_zero")


struct IncrBatchDimCtr:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return args[0]

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for IncrBatchDimCtr"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(decr_batch_dim_ctr(tangent))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        result: DeviceArray,
    ) raises -> DeviceArray:
        return incr_batch_dim_ctr(tangents[0])


fn incr_batch_dim_ctr(arg: DeviceArray) raises -> DeviceArray:
    if arg.batch_dim_ctr() == len(arg.shape()):
        raise "Cannot incr_batch_dim_ctr, batch_dim_ctr is already full"

    var res = register_unary_op[
        IncrBatchDimCtr.maxpr,
        IncrBatchDimCtr.vjp,
        IncrBatchDimCtr.jvp,
        IncrBatchDimCtr.eagerxpr,
    ](arg, "incr_batch_dim_ctr")
    res.batch_dim_ctr_(res.batch_dim_ctr() + 1)
    res.impl[].name = (
        "{" + String(res.batch_dim_ctr()) + "}" + res.impl[].name[3:]
    )
    return res


struct DecrBatchDimCtr:
    @staticmethod
    fn maxpr(
        args: List[Symbol], array: DeviceArray
    ) raises -> Symbol:
        return args[0]

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for DecrBatchDimCtr"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return List(incr_batch_dim_ctr(tangent))

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        result: DeviceArray,
    ) raises -> DeviceArray:
        return decr_batch_dim_ctr(tangents[0])


fn decr_batch_dim_ctr(arg: DeviceArray) raises -> DeviceArray:
    if arg.batch_dim_ctr() == 0:
        raise "Cannot decr_batch_dim_ctr, batch_dim_ctr is already 0"

    var res = register_unary_op[
        DecrBatchDimCtr.maxpr,
        DecrBatchDimCtr.vjp,
        DecrBatchDimCtr.jvp,
        DecrBatchDimCtr.eagerxpr,
    ](arg, "decr_batch_dim_ctr")
    res.batch_dim_ctr_(res.batch_dim_ctr() - 1)
    res.impl[].name = (
        "{" + String(res.batch_dim_ctr()) + "}" + res.impl[].name[3:]
    )
    return res
