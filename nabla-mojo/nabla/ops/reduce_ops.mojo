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

from collections import Dict

from nabla.core.device_array import DeviceArray, ArrayImpl
from nabla.ops.utils import register_any_op, RuntimeInfo
from nabla.ops.view_ops import squeeze, broadcast_to
from nabla.compiler.graph import Symbol, ops

alias BATCH_DIM_CTR = 0
alias ORIGINALshape = 2
alias AXES = 3
alias KEEP_DIM = 4
alias ORIGINAL_AXES = 5
alias ACT_ON_BATCH_DIMS = 6


struct Sum:
    @staticmethod
    fn maxpr(args: List[Symbol], array: DeviceArray) raises -> Symbol:
        var axes = array.impl[].runtime_info[AXES]
        var originalshape = array.impl[].runtime_info[ORIGINALshape]
        var symbol = args[0]

        for i in range(len(axes)):
            var axis = axes[i]
            symbol = ops.mean(symbol, axis) * originalshape[axis]

        return symbol

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Sum"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var act_on_batch_dims = True if array.impl[].runtime_info[
            ACT_ON_BATCH_DIMS
        ][0] == 1 else False
        var originalshape = array.impl[].runtime_info[ORIGINALshape]
        var target_shape = originalshape
        return List(
            broadcast_to(
                tangent,
                target_shape,
                act_on_batch_dims=act_on_batch_dims,
                expand_dims=False,
            )
        )

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var runtime_info = array.impl[].runtime_info
        var act_on_batch_dims = True if runtime_info[ACT_ON_BATCH_DIMS][
            0
        ] == 1 else False
        var axes = array.impl[].runtime_info[ORIGINAL_AXES]
        return sum(
            tangents[0],
            axes,
            keep_dim=True,
            act_on_batch_dims=act_on_batch_dims,
        )


fn sum(
    arg: DeviceArray,
    _axis: List[Int] = List[Int](),
    keep_dim: Bool = False,
    act_on_batch_dims: Bool = False,
) raises -> DeviceArray:
    var batch_dim_ctr = arg.batch_dim_ctr()
    if act_on_batch_dims:
        batch_dim_ctr = 0

    if arg.batch_dim_ctr() == len(arg.shape()):
        return arg

    var arg_shape = arg.shape()[batch_dim_ctr:]
    var axes = _axis

    if len(axes) == 0:
        for i in range(-len(arg_shape), 0):
            axes.append(i)
    else:
        for i in range(len(axes)):
            axes[i] = axes[i] if axes[i] < 0 else -len(arg_shape) + axes[i]
    sort(axes)

    var target_shape = arg.shape()[:batch_dim_ctr]
    for i in range(-len(arg_shape), 0):
        if i not in axes:
            target_shape.append(arg_shape[i])
        else:
            target_shape.append(1)

    if len(target_shape) == 0:
        target_shape.append(1)

    var runtime_info = RuntimeInfo(7)
    runtime_info[ORIGINAL_AXES] = _axis
    runtime_info[AXES] = axes
    runtime_info[ORIGINALshape] = arg_shape
    runtime_info[KEEP_DIM] = List[Int](1) if keep_dim else List[Int](0)
    runtime_info[BATCH_DIM_CTR] = List(batch_dim_ctr)
    runtime_info[ACT_ON_BATCH_DIMS] = List(1) if act_on_batch_dims else List(0)
    var name = "sum(" + axes.__str__() + ")"
    var res = register_any_op[Sum.maxpr, Sum.vjp, Sum.jvp, Sum.eagerxpr](
        List(arg), name, target_shape, runtime_info=runtime_info
    )
    if not keep_dim:
        if len(axes) == len(target_shape):
            axes = axes[:-1]
        for i in range(len(axes)):
            res = squeeze(res, List(axes[i]), act_on_batch_dims)

    return res
