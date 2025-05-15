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


from collections import Dict, Optional
from nabla.core.device_array import DeviceArray, ArrayImpl
from nabla.ops.view_ops import broadcast_to, unsqueeze
from nabla.ops.unary_ops import incr_batch_dim_ctr, decr_batch_dim_ctr
from nabla.api.utils import ExecutionContext
from nabla.api.utils import none
from nabla.compiler.graph import Symbol

fn generic_setup(args: List[DeviceArray], name: String) raises -> DeviceArray:
    var dtype = args[0].impl[].spec.dtype()
    var diffable = False
    var execution_context = Optional[ExecutionContext](None)
    var batch_dim_ctr = 0
    var arg_string: String = ""

    for arg in args:
        arg_string += arg[].dtype().__str__() + arg[].shape().__str__() + ","
        diffable = diffable or arg[].impl[]._diffable
        if arg[].impl[].spec.dtype() != dtype:
            raise "DType mismatch in arguments when registering op:" + name
        if arg[].batch_dim_ctr() > batch_dim_ctr:
            batch_dim_ctr = arg[].batch_dim_ctr()
        if arg[].impl[].execution_context:
            execution_context = arg[].impl[].execution_context

    var res = DeviceArray(
        shape=List(0),
        dtype=dtype,
        requires_pullback=diffable,
        execution_context=execution_context,
        name="{" + String(batch_dim_ctr) + "}" + name + "(" + arg_string + ")",
    )
    res.batch_dim_ctr_(batch_dim_ctr)

    for arg in args:
        res.impl[]._args.append(arg[].impl)

    return res


fn register_any_op[
    maxpr: fn (
        List[Symbol], DeviceArray
    ) raises -> Symbol,
    vjp: fn (List[DeviceArray], DeviceArray, DeviceArray) raises -> List[
        DeviceArray
    ],
    jvp: fn (
        List[DeviceArray], List[DeviceArray], DeviceArray
    ) raises -> DeviceArray,
    eagerxpr: fn (mut DeviceArray, List[DeviceArray]) raises -> None,
](
    args: List[DeviceArray],
    name: String,
    targetshape: List[Int],
    runtime_info: List[List[Int]] = List[List[Int]](),
) raises -> DeviceArray:
    var res = generic_setup(args, name)
    res.shape_(targetshape)
    res.impl[].runtime_info = runtime_info

    res.impl[]._maxpr = maxpr
    res.impl[]._vjp = vjp
    res.impl[]._jvp = jvp
    res.impl[]._eagerxpr = eagerxpr

    var arg_shapes: String = ""
    for arg in args:
        arg_shapes += arg[].shape().__str__() + ", "

    # print(
    #     "   ",
    #     res.impl[].name,
    #     targetshape.__str__(),
    #     " args:",
    #     arg_shapes,
    #     "batch_dim_ctr:",
    #     res.batch_dim_ctr(),
    # )

    return res


fn get_broadcasted_axis(
    argshape: List[Int], targetshape: List[Int]
) raises -> List[Int]:
    var broadcasted_axis = List[Int]()
    var rank = len(targetshape)
    var i = len(argshape) - 1
    var j = len(targetshape) - 1
    while j >= 0:
        if i >= 0 and argshape[i] == targetshape[j]:
            i -= 1
            j -= 1
        elif i >= 0 and argshape[i] == 1 and targetshape[j] > 1:
            broadcasted_axis.append(-rank + j)
            i -= 1
            j -= 1
        elif i < 0:
            broadcasted_axis.append(-rank + j)
            j -= 1
        else:
            raise "Invalid broadcast, trying to broadcast from " + argshape.__str__() + " to " + targetshape.__str__()

    return broadcasted_axis


fn get_broadcastedshape(
    arg0: List[Int], arg1: List[Int], right_offset: Int = 0
) raises -> List[Int]:
    var newshape = List[Int]()
    var i = len(arg0) - 1 - right_offset
    var j = len(arg1) - 1 - right_offset

    # the ignored shape elements must be set manually
    for _ in range(right_offset):
        newshape.append(-1)

    if len(arg0) == 0:
        return arg1
    if len(arg1) == 0:
        return arg0

    while i >= 0 or j >= 0:
        if i >= 0 and j >= 0:
            if arg0[i] == arg1[j]:
                newshape.append(arg0[i])
                i -= 1
                j -= 1
            elif arg0[i] == 1:
                newshape.append(arg1[j])
                j -= 1
                i -= 1
            elif arg1[j] == 1:
                newshape.append(arg0[i])
                i -= 1
                j -= 1
            else:
                raise "Invalid broadcast, when finding the brshape for: " + arg0.__str__() + " and " + arg1.__str__()
        elif i >= 0:
            newshape.append(arg0[i])
            i -= 1
        elif j >= 0:
            newshape.append(arg1[j])
            j -= 1

    newshape.reverse()

    return newshape


fn register_binary_op[
    maxpr: fn (
        List[Symbol], DeviceArray
    ) raises -> Symbol,
    vjp: fn (List[DeviceArray], DeviceArray, DeviceArray) raises -> List[
        DeviceArray
    ],
    jvp: fn (
        List[DeviceArray], List[DeviceArray], DeviceArray
    ) raises -> DeviceArray,
    eagerxpr: fn (mut DeviceArray, List[DeviceArray]) raises -> None,
](
    read _arg0: DeviceArray,
    read _arg1: DeviceArray,
    name: String,
) raises -> DeviceArray:
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
    )
    var new_shape = res_batch_dim + res_true_dim

    arg0 = broadcast_to(arg0, res_true_dim)
    arg1 = broadcast_to(arg1, res_true_dim)
    arg0 = broadcast_to(arg0, new_shape, act_on_batch_dims=True)
    arg1 = broadcast_to(arg1, new_shape, act_on_batch_dims=True)

    return register_any_op[maxpr, vjp, jvp, eagerxpr](
        List(arg0, arg1),
        name,
        new_shape,
    )


fn register_unary_op[
    maxpr: fn (
        List[Symbol], DeviceArray
    ) raises -> Symbol,
    vjp: fn (List[DeviceArray], DeviceArray, DeviceArray) raises -> List[
        DeviceArray
    ],
    jvp: fn (
        List[DeviceArray], List[DeviceArray], DeviceArray
    ) raises -> DeviceArray,
    eagerxpr: fn (mut DeviceArray, List[DeviceArray]) raises -> None,
](arg: DeviceArray, name: String,) raises -> DeviceArray:
    return register_any_op[maxpr, vjp, jvp, eagerxpr](
        List(arg), name, arg.shape()
    )


fn RuntimeInfo(cap: Int) raises -> List[List[Int]]:
    var runtime_info = List[List[Int]]()
    for _ in range(cap):
        runtime_info.append(List[Int]())
    return runtime_info
