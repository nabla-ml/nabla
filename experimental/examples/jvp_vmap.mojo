# ===----------------------------------------------------------------------=== #
# Endia 2025
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

import endia


def foo(args: List[endia.Array]) -> List[endia.Array]:
    x = args[0]
    y = args[1]
    return [endia.sin(x) + x**2 + y**2, endia.cos(y) + y * x]


def test_jvp_vmap():
    var x = endia.arange((2, 3)) + 2
    var y = endia.arange((2, 3)) + 3
    var v = endia.arange((3, 2, 3))
    var w = endia.arange((3, 2, 3))

    # Step 1: Compute the gradient using VJP.
    fn reg_jvp(args: List[endia.Array]) raises -> List[endia.Array]:
        var primals = args[: len(args) // 2]
        var tangents = args[len(args) // 2 :]
        return endia.jvp(foo, primals, tangents)[1]  # This extracts f'(x)

    jvp_vmapped = endia.vmap(reg_jvp, in_axes=[endia.none, endia.none, 0, 0])
    jvp_result = jvp_vmapped([x, y, v, w])

    print("First Order JVP:")
    # print(endia.xpr(grad_fn)([x,]))
    print(jvp_result[0])
    print(jvp_result[1])
