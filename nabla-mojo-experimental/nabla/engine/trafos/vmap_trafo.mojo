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
from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    get_full_trace_recursively,
    Callable,
    callable,
)
from nabla.api.utils import none


fn adapt_to_in_axis(
    mut arg: Array, in_axis: Int, batch_size: Int
) raises -> Array:
    if in_axis == none:
        var res = unsqueeze(arg, List(0))
        if batch_size > 1:
            var batch_dim_ctr = res.device_array[].impl[]._batch_dim_ctr
            var shape = res.device_array[].impl[].shape[batch_dim_ctr:]
            shape[0] = batch_size
            res = broadcast_to(res, shape)

        res = incr_batch_dim_ctr(res)
        return res
    else:
        var axis = in_axis
        var rank = len(arg.device_array[].impl[].shape)
        if axis < 0:
            axis = rank + axis
        if axis >= rank:
            raise "Error: in_axis in adapt_to_in_axis is out of bounds."

        if axis == 0:
            return incr_batch_dim_ctr(arg)
        elif axis > 0:
            var res = transpose(arg, axis, 0)
            return incr_batch_dim_ctr(res)
        else:
            raise "Error: Invalid in_axis."


fn adapt_to_out_axis(mut arg: Array, out_axis: Int) raises -> Array:
    if out_axis == none:
        var idx = arg.device_array[].impl[]._batch_dim_ctr - 1
        var res = decr_batch_dim_ctr(arg)
        if arg.device_array[].impl[].shape[idx] == 1:
            return squeeze(res, List(0))
        return res
    else:
        var axis = out_axis
        var rank = len(arg.device_array[].impl[].shape)
        if axis < 0:
            axis = rank + axis
        elif axis >= rank:
            raise "Error: out_axis in adapt_to_out_axis is out of bounds."

        if axis == 0:
            var res = decr_batch_dim_ctr(arg)
            return res
        elif axis > 0:
            var res = decr_batch_dim_ctr(arg)
            res = transpose(res, axis, 0)
            return res
        else:
            raise "Error: Invalid out_axis."


fn vmap_start_rule(
    mut args: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    var in_axes = meta["in_axes"] if "in_axes" in meta else List[Int]()
    if len(in_axes) == 0:
        for _ in range(len(args)):
            in_axes.append(0)

    meta["in_axes"] = in_axes

    # get unified batch dimension, check if is either none or equal and check if we have at least one non-none in_axis
    var batch_size = -1
    var num_none_dims = 0
    for i in range(len(in_axes)):
        var axis = in_axes[i]
        if axis == none:
            num_none_dims += 1
        else:
            var batch_dim_ctr = args[i].batch_dim_ctr()
            var size = args[i].shape()[axis + batch_dim_ctr]
            if batch_size != -1 and size != batch_size:
                # batch_siez has been set before and was a different value -> Error
                raise "Error in in_axes: ones axis has size " + size.__str__() + " but previous batch dimension size was defined as " + batch_size.__str__()
            if batch_size == -1:
                batch_size = size

    # adapt batch_dim_ctrs for all args
    var adapted_args = List[Array]()
    for i in range(len(args)):
        var arg = args[i]
        var axis = in_axes[i] if len(in_axes) > i else 0
        var adapted_arg = adapt_to_in_axis(arg, axis, batch_size)
        adapted_args.append(adapted_arg)

    return adapted_args


fn vmap_end_rule(
    mut args: List[Array],
    mut res: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    var out_axes = meta["out_axes"] if "out_axes" in meta else List[Int]()
    if len(out_axes) == 0:
        for _ in range(len(res)):
            out_axes.append(0)

    meta["out_axes"] = out_axes

    var adapted_res = List[Array]()
    for i in range(len(res)):
        var axis = out_axes[i] if len(out_axes) > i else 0
        var new_res = res[i]

        new_res = adapt_to_out_axis(new_res, axis)
        adapted_res.append(new_res)

    return adapted_res
