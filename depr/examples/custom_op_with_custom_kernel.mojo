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

import nabla
from nabla.core.device_array import DeviceArray
from nabla.ops.utils import register_unary_op
from nabla.compiler.graph import Symbol


struct CustomOp:
    @staticmethod
    fn maxpr(args: List[Symbol], array: DeviceArray) raises -> Symbol:
        return nabla.compiler.graph.ops.custom["custom_op"](
            args[0], args[0].type()
        )

    @staticmethod
    fn eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for CustomOp"

    @staticmethod
    fn vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        return [
            custom_op(tangent),
        ]

    @staticmethod
    fn jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        return custom_op(tangents[0])


fn custom_op(arg: DeviceArray) raises -> DeviceArray:
    return register_unary_op[
        CustomOp.maxpr, CustomOp.vjp, CustomOp.jvp, CustomOp.eagerxpr
    ](
        arg,
        name="custom_op",
        custom_kernel_path=String("./examples/custom_kernels/kernels.mojopkg"),
    )


fn custom_op(arg: nabla.Array) raises -> nabla.Array:
    return nabla.Array(custom_op(arg.device_array[]))


fn test_custom_op_with_custom_kernel() raises -> None:
    # Test the custom op with a custom kernel
    x = nabla.ndarange((2, 3))
    res = custom_op(x)
    print(res)
