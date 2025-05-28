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

from utils import Variant

from nabla.api.array import Array
from nabla.core.device_array import DeviceArray, zeros_like
from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    get_full_trace_recursively_jvp,
    Callable,
    callable,
)
from nabla.api.utils import none


fn jvp_call(
    meta: TrafoMeta,
    args: List[Array],
) raises -> List[Array]:
    return args[: len(args) // 2]


fn jvp_end_rule(
    mut _args: List[Array],
    mut res: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    meta["num_res"] = List(len(res))

    var args = _args[: len(_args) // 2]
    var tangents = _args[len(_args) // 2 :]

    for i in range(len(args)):
        args[i].device_array[].impl[].tangents = List(
            tangents[i].device_array[].impl
        )

    for arg in args:
        arg[].device_array[].impl[]._compute_jvp = True
        arg[].device_array[].impl[].tangents[-1][]._compute_jvp = True

    var trace = List[DeviceArray]()
    for i in range(len(res)):
        get_full_trace_recursively_jvp(trace, res[i].device_array[])

    for i in range(len(trace)):
        var array = trace[i]

        if not array.impl[]._jvp:
            raise "Error in JVP end rule: The operation " + array.impl[].name + " does not have the expected jvp rule defined!"

        var jvp_rule = array.impl[]._jvp.value()
        var primals = List[DeviceArray]()
        var tangents = List[DeviceArray]()
        if len(array.args()) == 1 and not array.args()[0].has_tangent():
            continue

        for arg in array.args():
            var primal = arg[]
            primals.append(primal)
            if len(primal.impl[].tangents) == 0:
                tangent = zeros_like(primal)
                tangents.append(tangent)
            else:
                var tangent = DeviceArray(primal.impl[].tangents[-1])
                tangents.append(tangent)

        var array_tangent = jvp_rule(primals, tangents, array)
        array.impl[].tangents = List(array_tangent.impl)
        array.impl[]._compute_jvp = False

    var res_tangents = List[Array]()
    for i in range(len(res)):
        if len(res[i].device_array[].impl[].tangents) == 0:
            res_tangents.append(Array(zeros_like(res[i].device_array[])))
        else:
            res_tangents.append(
                Array(DeviceArray(res[i].device_array[].impl[].tangents[0]))
            )

    for array in trace:
        array[].impl[].tangents.clear()

    res = res + res_tangents
    for i in range(len(args)):
        _args[i].device_array[].impl[]._compute_jvp = False
        _args[i].device_array[].impl[].tangents.clear()
    for i in range(len(res)):
        res[i].device_array[].impl[].tangents.clear()

    return res
