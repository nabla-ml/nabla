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

from nabla.api.array import Array
from nabla.core.device_array import DeviceArray, zeros_like
from nabla.engine.trafos.vjp_trafo import cotangent
from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    cotangent_with_remat,
    Callable,
    callable,
)
from nabla.api.utils import none
from nabla.api.array import zeros


fn jacrev_start_rule(
    mut args: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    return args


fn jacrev_call(
    meta: TrafoMeta,
    args: List[Array],
) raises -> List[Array]:
    var primals = args
    for primal in primals:
        primal[].requires_pullback_(True)
    return primals


fn jacrev_end_rule(
    mut args: List[Array],
    mut res: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    var primals = args

    sizes, tangents = std_basis(res)

    for i in range(len(tangents)):
        tangents[i] = incr_batch_dim_ctr(tangents[i])

    for primal in primals:
        primal[].requires_pullback_(True)

    if len(tangents) != len(res):
        raise "Error in jacrev_end_rule: Number of tangents does not match the number of outputs. len(tangents) = " + len(
            tangents
        ).__str__() + " vs. len(res) = " + len(
            res
        ).__str__()

    var outputs = List[DeviceArray]()
    for i in range(len(res)):
        var device_array = res[i].device_array
        var tangent = tangents[i].device_array
        device_array[].impl[].cotangent = List(tangent[].impl)
        outputs.append(device_array[])

    if len(meta["with_remat"]) == 0:
        raise "Error in jacrev_end_rule: with_remat is not defined in meta."
    var remat = meta["with_remat"][0]

    _ = cotangent_with_remat(outputs) if remat else cotangent(outputs)
    var grads = List[Array]()
    for i in range(len(primals)):
        if len(primals[i].device_array[].impl[].cotangent) == 0:
            var shape = List(sizes[i]) + primals[i].shape()
            var empty_grad = zeros(shape, primals[i].dtype())
            empty_grad.batch_dim_ctr_(primals[i].batch_dim_ctr())
            grads.append(empty_grad)
        else:
            var grad = primals[i].cotangent()
            grads.append(decr_batch_dim_ctr(grad))

    var cotangents = List[Array]()

    var splits = List[List[Array]]()
    for j in range(len(grads)):
        splits.append(split(grads[j], sizes=sizes, axis=0))

    var values = outputs
    for j in range(len(values)):
        for i in range(len(args)):
            var grad = splits[i][j]
            var batch_dim_ctr_arg = args[i].batch_dim_ctr()
            batch_dim_ctr_arg = (
                batch_dim_ctr_arg if batch_dim_ctr_arg != none else 0
            )
            var batch_dim_ctr_out = values[j].batch_dim_ctr()
            batch_dim_ctr_out = (
                batch_dim_ctr_out if batch_dim_ctr_out != none else 0
            )
            var arg_shape = args[i].shape()[batch_dim_ctr_arg:]
            var out_shape = values[j].shape()[batch_dim_ctr_out:]
            if len(arg_shape) == 1 and arg_shape[0] == 1:
                arg_shape.clear()
            elif len(out_shape) == 1 and out_shape[0] == 1:
                out_shape.clear()

            var shape = out_shape + arg_shape
            reshaped_grad = grad.reshape(shape)
            cotangents.append(reshaped_grad)

    for array in primals:
        array[].device_array[].impl[].cotangent.clear()
        array[].requires_pullback_(False)

    return cotangents
