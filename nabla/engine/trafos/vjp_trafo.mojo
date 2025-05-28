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

from memory import ArcPointer
from utils import Variant
from nabla.api.array import Array
from nabla.core.device_array import DeviceArray, zeros_like
from nabla.api.array import ones
from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    get_full_trace_recursively,
    Callable,
    callable,
)


fn compute_cotangent(
    mut array: DeviceArray, mut cotangents: List[DeviceArray]
) raises -> None:
    if not array.impl[]._vjp:
        raise "_vjp not found for array:" + array.impl[].name
    var _vjp = array.impl[]._vjp.value()

    var primals = List[DeviceArray]()
    for arg in array.args():
        primals.append(arg[])

    if len(array.impl[].cotangent) == 0:
        raise "DeviceArray has no cotangent, id = " + String(
            array.id()
        ) + " and name = " + array.impl[].name
    var cotangent = DeviceArray(array.impl[].cotangent[0])

    var tangents = _vjp(primals, cotangent, array)

    for j in range(len(primals)):
        var primal = primals[j]
        var primal_cotangent = tangents[j]
        if primal.impl[]._diffable:
            if len(primal.impl[].cotangent) > 0:
                primal_cotangent = primal_cotangent + DeviceArray(
                    primal.impl[].cotangent[0]
                )

            primal.impl[].cotangent = List(primal_cotangent.impl)

    for j in range(len(primals)):
        var primal = primals[j]
        if primal.impl[].requires_pullback:
            cotangents.append(primal.cotangent())


fn cotangent(
    outs: List[DeviceArray], keep_graph: Bool = True
) raises -> List[DeviceArray]:
    var trace = List[DeviceArray]()

    for output in outs:
        var parent = output[]
        get_full_trace_recursively(trace, parent)

    var cotangents = List[DeviceArray]()

    for i in range(len(trace) - 1, -1, -1):
        var array = trace[i]
        array.visited_(False)

        if (
            len(array.args()) == 0
            or not array.impl[]._diffable
            or array.impl[].requires_pullback
        ):
            continue

        compute_cotangent(array, cotangents)
        array.impl[].cotangent.clear()

    if not keep_graph:
        for cotangent in cotangents:
            cotangent[].requires_pullback_(False)

    return cotangents


fn reset_visited(mut trace: List[DeviceArray]) raises -> None:
    for array in trace:
        array[].visited_(False)


fn cotangent_with_remat(
    outs: List[DeviceArray], keep_graph: Bool = True
) raises -> List[DeviceArray]:
    var trace = List[DeviceArray]()

    for output in outs:
        var parent = output[]
        get_full_trace_recursively(trace, parent)

    reset_visited(trace)

    for array in trace:
        if (
            not array[].impl[].is_checkpoint
            and (not array[].impl[].requires_pullback)
            and array[].impl[]._diffable
            and (not array[].is_tmp_output())
        ):
            var dual_args = array[].args()

            for i in range(len(dual_args)):
                var arg = dual_args[i]
                if len(arg.impl[]._dual) == 1:
                    dual_args[i] = arg.dual()

            var dual = DeviceArray(ArcPointer(array[].impl[]))
            dual.name_("dual_" + array[].impl[].name)
            dual.args_(dual_args)
            array[].dual_(dual)

    var cotangents = List[DeviceArray]()

    for i in range(len(trace) - 1, -1, -1):
        var p_array = trace[i]
        var array = p_array

        if (
            len(array.args()) == 0
            or not array.impl[]._diffable
            or array.impl[].requires_pullback
        ):
            continue

        if len(array.impl[]._dual) == 1:
            array = DeviceArray(p_array.impl[]._dual[0])
            if (
                len(p_array.impl[].cotangent) == 1
                and len(array.impl[].cotangent) == 0
            ):
                array.impl[].cotangent = p_array.impl[].cotangent

        compute_cotangent(array, cotangents)
        array.impl[].cotangent.clear()
        p_array.impl[].cotangent.clear()

    for array in trace:
        array[].impl[]._dual.clear()

    if not keep_graph:
        for cotangent in cotangents:
            cotangent[].requires_pullback_(False)

    return cotangents


fn vjp_call(
    meta: TrafoMeta,
    args: List[Array],
) raises -> List[Array]:
    var num_primals = meta["num_primals"][0]
    var primals = args[:num_primals]
    for primal in primals:
        primal[].requires_pullback_(True)
    return primals


fn vjp_end_rule(
    mut args: List[Array],
    mut res: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    var outputs = List[DeviceArray]()
    var num_primals = meta["num_primals"][0]
    var primals = args[:num_primals]
    var tangents = args[num_primals:]

    for primal in primals:
        primal[].requires_pullback_(True)

    if len(tangents) != len(res):
        raise "Error in vjp_end_rule: Number of tangents does not match the number of outputs. len(tangents) = " + len(
            tangents
        ).__str__() + " vs. len(res) = " + len(
            res
        ).__str__()

    for i in range(len(res)):
        var device_array = res[i].device_array
        var tangent = tangents[i].device_array
        device_array[].impl[].cotangent = List(tangent[].impl)
        outputs.append(device_array[])

    if len(meta["with_remat"]) == 0:
        raise "Error in vjp_end_rule: with_remat is not defined in meta."
    var remat = meta["with_remat"][0]

    _ = cotangent_with_remat(outputs) if remat else cotangent(outputs)
    var cotangents = List[Array]()
    for primal in primals:
        cotangents.append(primal[].cotangent())

    for array in primals:
        array[].device_array[].impl[].cotangent.clear()
        array[].requires_pullback_(False)

    return cotangents


fn backward(
    array: DeviceArray, remat: Bool = False, keep_graph: Bool = False
) raises -> None:
    if len(array.impl[].shape) != 1 and array.impl[].shape[0] != 1:
        raise "Error: Backward only supports Scalar Outputs."
    var execution_context = array.impl[].execution_context
    var dtype = array.impl[].dtype
    var tangent = ones(
        (1,),
        dtype,
        requires_grad=False,
        execution_context=execution_context,
    )
    array.impl[].cotangent = List(tangent.device_array[].impl)

    if remat:
        _ = cotangent_with_remat(List(array), keep_graph)
    else:
        _ = cotangent(List(array), keep_graph)
