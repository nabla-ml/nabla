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


fn test_jacrev_jacfwd() raises:
    var x = nabla.arange((2, 3))

    def foo(args: List[nabla.Array]) -> List[nabla.Array]:
        var x = args[0]
        return [
            nabla.sum(x * x * x),
        ]

    fn d1foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return nabla.jacrev(foo)(args)

    fn d2foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return nabla.jacfwd(d1foo)(args)

    fn d3foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return nabla.jacfwd(d2foo)(args)

    var d1foo_res = d1foo(
        [
            x,
        ]
    )
    # print(nabla.xpr(d1foo)([x,]))
    print("\nd1st Derivative 1 (d1foo):")
    print((d1foo_res[0]))

    var d2foo_res = d2foo(
        [
            x,
        ]
    )
    # print(nabla.xpr(d2foo)([x,]))
    print("\n2nd Derivative 1 (d2foo):")
    print(nabla.sum(d2foo_res[0]))

    var d3foo_res = d3foo(
        [
            x,
        ]
    )
    # print(nabla.xpr(d3foo)([x,]))
    print("\n3rd Derivative 1 (d3foo):")
    print(nabla.sum(d3foo_res[0]))
