# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from nabla.api.array import Array
from nabla.core.device_array import DeviceArray, zeros_like
from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    get_full_trace_recursively_jvp,
    Callable,
    callable,
)
from nabla.api.ops import incr_batch_dim_ctr, decr_batch_dim_ctr
from nabla.api.utils import none


fn jacfwd_end_rule(
    mut _args: List[Array],
    mut res: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    meta["num_res"] = List(len(res))

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
