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
    return [x * x, y * y]


def test_vjp_vmap():
    var x = endia.arange((3, 2, 3)) + 2
    var y = endia.arange((3, 2, 3)) + 3

    # Step 1: Compute the gradient using VJP.
    def vjp_vmapped(args: List[endia.Array]) -> List[endia.Array]:
        _, vjp_fn = endia.vjp(endia.vmap(foo), args)
        return vjp_fn([endia.ones((3, 2, 3)), endia.ones((3, 2, 3))])

    vjp_result = vjp_vmapped([x, y])

    print("First Order VJP:")
    # print(endia.xpr(grad_fn)([x,]))
    print(vjp_result[0])
    print(vjp_result[1])
