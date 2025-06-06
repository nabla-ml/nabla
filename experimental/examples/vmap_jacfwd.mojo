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


fn test_vmap_jacfwd() raises:
    var x = endia.arange((3, 2, 3)) + 2
    var y = endia.arange((3, 2, 3)) + 3

    def foo(args: List[endia.Array]) -> List[endia.Array]:
        var x = args[0]
        var y = args[1]
        var res = [endia.sin(x) + x**2 + y**2, endia.cos(y) + y * x]
        return res

    var jacobian = endia.jacfwd(foo)
    jacobian_result = jacobian([x, y])
    # print(endia.xpr(jacobian)([x,y]))

    print("\nJacobian 1:")
    print(jacobian_result[0])
    print("\nJacobian 2:")
    print(jacobian_result[1])
    print("\nJacobian 3:")
    print(jacobian_result[2])
    print("\nJacobian 4:")
    print(jacobian_result[3])
