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


fn test_jacrev_jacfwd_jacfwd() raises:
    fn foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return [
            nabla.sum(nabla.sin(args[0] * args[1])),
            nabla.sum(nabla.cos(args[0] * args[1])),
        ]

    var foo_vmapped = nabla.vmap(foo)
    var foo_d1 = nabla.jacrev(foo_vmapped)
    var foo_d2 = nabla.jacfwd(foo_d1)
    var foo_d3 = nabla.jacfwd(foo_d2)

    var args = [nabla.arange((2, 3)), nabla.arange((2, 3))]

    var res = foo(args)
    # print(nabla.xpr(foo)(args))
    # print(res[0])
    print("foo checksum: ", nabla.sum(res[0]))  # , nabla.sum(res[1]))

    var d1 = foo_d1(args)
    # print(nabla.xpr(foo_d1)(args))
    # print(d1[0])
    print(
        "foo_d1 checksum: ", nabla.sum(d1[0])
    )  # , nabla.sum(d1[1]), nabla.sum(d1[2]), nabla.sum(d1[3]))

    var d2 = foo_d2(args)
    # print(nabla.xpr(foo_d2)(args))
    # print(d2[0])
    print(
        "foo_d2 checksum: ", nabla.sum(d2[0])
    )  # , nabla.sum(d2[1]), nabla.sum(d2[2]), nabla.sum(d2[3]), nabla.sum(d2[4]), nabla.sum(d2[5]), nabla.sum(d2[6]), nabla.sum(d2[7]))

    var d3 = foo_d3(args)
    print(
        "foo_d3 checksum: ", nabla.sum(d3[0])
    )  # , nabla.sum(d3[1]), nabla.sum(d3[2]), nabla.sum(d3[3]), nabla.sum(d3[4]), nabla.sum(d3[5]), nabla.sum(d3[6]), nabla.sum(d3[7]), nabla.sum(d3[8]), nabla.sum(d3[9]), nabla.sum(d3[10]), nabla.sum(d3[11]), nabla.sum(d3[12]), nabla.sum(d3[13]), nabla.sum(d3[14]), nabla.sum(d3[15]))
