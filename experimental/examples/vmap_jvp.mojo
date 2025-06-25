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


def foo(args: List[nabla.Array]) -> List[nabla.Array]:
    x = args[0]
    y = args[1]
    return [nabla.sin(x) + x**2 + y**2, nabla.cos(y) + y * x]


def test_vmap_jvp():
    var x = nabla.ndarange((3, 2, 3)) + 2
    var y = nabla.ndarange((3, 2, 3)) + 3
    var v = nabla.ndarange((3, 2, 3))
    var w = nabla.ndarange((3, 2, 3))

    # Step 1: Compute the gradient using VJP.
    fn jvp_vmapped(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var primals = args[: len(args) // 2]
        var tangents = args[len(args) // 2 :]
        # print("Start VMAP")
        var foo_vmap = nabla.vmap(foo)
        # print("End VMAP")
        return nabla.jvp(foo_vmap, primals, tangents)[1]  # This extracts f'(x)

    jvp_result = jvp_vmapped([x, y, v, w])

    print("First Order JVP:")
    # print(nabla.xpr(grad_fn)([x,]))
    print(jvp_result[0])
    print(jvp_result[1])
