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
import .. ops as ops
from nabla.core.utils import ShapeType, getshape
from nabla.core.device_array import DeviceArray
import ..engine.trafos.vjp_trafo as vjp_trafo


fn add(x: Array, y: Array) raises -> Array:
    return Array(ops.binary_ops.add(x.device_array[], y.device_array[]))


fn mul(x: Array, y: Array) raises -> Array:
    return Array(ops.binary_ops.mul(x.device_array[], y.device_array[]))


fn sub(x: Array, y: Array) raises -> Array:
    return Array(ops.binary_ops.sub(x.device_array[], y.device_array[]))


fn div(x: Array, y: Array) raises -> Array:
    return Array(ops.binary_ops.div(x.device_array[], y.device_array[]))


fn matmul(x: Array, y: Array) raises -> Array:
    return Array(ops.binary_ops.matmul(x.device_array[], y.device_array[]))


fn gt(x: Array, y: Array) raises -> Array:
    return Array(ops.binary_ops.gt(x.device_array[], y.device_array[]))


fn pow(x: Array, y: Array) raises -> Array:
    return Array(ops.binary_ops.pow(x.device_array[], y.device_array[]))


fn sum(
    x: Array,
    axis: List[Int] = List[Int](),
    keep_dim: Bool = False,
    act_on_batch_dims: Bool = False,
) raises -> Array:
    return Array(
        ops.reduce_ops.sum(x.device_array[], axis, keep_dim, act_on_batch_dims)
    )


fn sin(x: Array) raises -> Array:
    return Array(ops.unary_ops.sin(x.device_array[]))


fn cast(x: Array, dtype: DType) raises -> Array:
    return Array(ops.unary_ops.cast(x.device_array[], dtype))


fn negate(x: Array) raises -> Array:
    return Array(ops.unary_ops.negate(x.device_array[]))


fn cos(x: Array) raises -> Array:
    return Array(ops.unary_ops.cos(x.device_array[]))


fn relu(x: Array) raises -> Array:
    return Array(ops.unary_ops.relu(x.device_array[]))


fn log(x: Array) raises -> Array:
    return Array(ops.unary_ops.log(x.device_array[]))


fn gt_zero(x: Array) raises -> Array:
    return Array(ops.unary_ops.gt_zero(x.device_array[]))


fn incr_batch_dim_ctr(x: Array) raises -> Array:
    return Array(ops.unary_ops.incr_batch_dim_ctr(x.device_array[]))


fn decr_batch_dim_ctr(x: Array) raises -> Array:
    return Array(ops.unary_ops.decr_batch_dim_ctr(x.device_array[]))


fn permute(arg: Array, perm: List[Int]) raises -> Array:
    return Array(ops.view_ops.permute(arg.device_array[], perm))


fn transpose(arg: Array, x: Int, y: Int) raises -> Array:
    return Array(ops.view_ops.transpose(arg.device_array[], x, y))


fn reshape(x: Array, shape: ShapeType) raises -> Array:
    return Array(ops.view_ops.reshape(x.device_array[], getshape(shape)))


fn flatten(x: Array) raises -> Array:
    return Array(ops.view_ops.flatten(x.device_array[]))


fn broadcast_to(
    x: Array, shape: ShapeType, act_on_batch_dims: Bool = False
) raises -> Array:
    return Array(
        ops.view_ops.broadcast_to(
            x.device_array[], getshape(shape), act_on_batch_dims
        )
    )


fn stack(args: List[Array], axis: Int = 0) raises -> Array:
    var device_arrays = List[DeviceArray]()
    for arg in args:
        device_arrays.append(arg[].device_array[])
    return Array(ops.view_ops.stack(device_arrays, axis))


fn array_slice(arg: Array, slices: List[Slice]) raises -> Array:
    return Array(ops.view_ops.array_slice(arg.device_array[], slices))


fn concat(args: List[Array], axis: Int = 0) raises -> Array:
    var device_arrays = List[DeviceArray]()
    for arg in args:
        device_arrays.append(arg[].device_array[])
    return Array(ops.view_ops.concat(device_arrays, axis))


fn split(arg: Array, sizes: List[Int], axis: Int) raises -> List[Array]:
    var device_arrays = ops.view_ops.split(arg.device_array[], sizes, axis)
    var results = List[Array]()
    for device_array in device_arrays:
        results.append(Array(device_array[]))
    return results


fn squeeze(
    arg: Array, axis: List[Int], act_on_batch_dims: Bool = False
) raises -> Array:
    return Array(
        ops.view_ops.squeeze(arg.device_array[], axis, act_on_batch_dims)
    )


fn unsqueeze(
    arg: Array, axes: List[Int], act_on_batch_dims: Bool = False
) raises -> Array:
    return Array(
        ops.view_ops.unsqueeze(arg.device_array[], axes, act_on_batch_dims)
    )


fn backward(array: Array, remat: Bool = False) raises -> None:
    vjp_trafo.backward(array.device_array[], remat)
