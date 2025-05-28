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

from nabla.api.array import Array, zeros
from nabla.core.device_array import DeviceArray, zeros_like
from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    get_full_trace_recursively_jvp,
    cotangent_with_remat,
    Callable,
    callable,
)
from nabla.api.ops import incr_batch_dim_ctr, decr_batch_dim_ctr
from nabla.api.utils import none
from nabla.engine.trafos.vjp_trafo import cotangent


fn grad_call(
    meta: TrafoMeta,
    args: List[Array],
) raises -> List[Array]:
    var primals = args
    for primal in primals:
        primal[].requires_pullback_(True)
    return primals


fn grad_end_rule(
    mut _args: List[Array],
    mut res: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    meta["num_res"] = List(len(res))

    var num_elements_args = 0
    var num_elements_res = 0
    for arg in _args:
        var num_elements = 1
        for dim in arg[].shape()[arg[].batch_dim_ctr() :]:
            num_elements *= dim[]
        num_elements_args += num_elements
    for res in res:
        var num_elements = 1
        for dim in res[].shape()[res[].batch_dim_ctr() :]:
            num_elements *= dim[]
        num_elements_res += num_elements

    if num_elements_args > num_elements_res:
        var primals = _args
        var args = _args

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

    else:
        var args = _args
        sizes, tangents = std_basis(args)

        for i in range(len(tangents)):
            tangents[i] = incr_batch_dim_ctr(tangents[i])

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
                raise "Error in JVP end rule: The operation " + array.impl[].name + " does not have the expected jacfwd rule defined!"

            var jacfwd_rule = array.impl[]._jvp.value()
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

            var array_tangent = jacfwd_rule(primals, tangents, array)
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

        for i in range(len(res_tangents)):
            res_tangents[i] = decr_batch_dim_ctr(res_tangents[i])

        var grads = res_tangents
        tangents = List[Array]()
        var splits = List[List[Array]]()

        for grad in grads:
            splits.append(split(grad[], sizes=sizes, axis=0))

        var values = res

        for i in range(len(grads)):
            for j in range(len(splits[i])):
                var value = values[i]
                var arg = args[j]
                var grad = splits[i][j]

                var batch_dim_ctr_arg = arg.batch_dim_ctr()
                batch_dim_ctr_arg = (
                    batch_dim_ctr_arg if batch_dim_ctr_arg != none else 0
                )
                var batch_dim_ctr_out = value.batch_dim_ctr()
                batch_dim_ctr_out = (
                    batch_dim_ctr_out if batch_dim_ctr_out != none else 0
                )
                var arg_shape = arg.shape()[batch_dim_ctr_arg:]
                var out_shape = value.shape()[batch_dim_ctr_out:]
                if len(arg_shape) == 1 and arg_shape[0] == 1:
                    arg_shape.clear()
                elif len(out_shape) == 1 and out_shape[0] == 1:
                    out_shape.clear()

                var shape = arg_shape + out_shape
                reshaped_grad = grad.reshape(shape)
                var perm_axes = List[Int]()
                for k in range(len(out_shape)):
                    perm_axes.append(k + len(arg_shape))
                for k in range(len(arg_shape)):
                    perm_axes.append(k)
                var permuted_grad = permute(reshaped_grad, perm_axes)
                tangents.append(permuted_grad)

        for i in range(len(args)):
            _args[i].device_array[].impl[]._compute_jvp = False
            _args[i].device_array[].impl[].tangents.clear()
        for i in range(len(tangents)):
            tangents[i].device_array[].impl[].tangents.clear()

        return tangents
