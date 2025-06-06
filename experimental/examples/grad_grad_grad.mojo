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


fn test_grad_grad_grad() raises:
    fn foo(args: List[endia.Array]) raises -> List[endia.Array]:
        return [
            endia.sum(endia.sin(args[0] * args[1])),
            endia.sum(endia.cos(args[0] * args[1])),
        ]

    var foo_vmapped = endia.vmap(foo)
    var foo_d1 = endia.grad(foo_vmapped)
    var foo_d2 = endia.grad(foo_d1)
    var foo_d3 = endia.grad(foo_d2)

    var args = [endia.arange((2, 3)), endia.arange((2, 3))]

    var res = foo(args)
    # print(endia.xpr(foo)(args))
    # print(res[0])
    print("foo checksum: ", endia.sum(res[0]))  # , endia.sum(res[1]))

    var d1 = foo_d1(args)
    # print(endia.xpr(foo_d1)(args))
    # print(d1[0])
    print(
        "foo_d1 checksum: ", endia.sum(d1[0])
    )  # , endia.sum(d1[1]), endia.sum(d1[2]), endia.sum(d1[3]))

    var d2 = foo_d2(args)
    # print(endia.xpr(foo_d2)(args))
    # print(d2[0])
    print(
        "foo_d2 checksum: ", endia.sum(d2[0])
    )  # , endia.sum(d2[1]), endia.sum(d2[2]), endia.sum(d2[3]), endia.sum(d2[4]), endia.sum(d2[5]), endia.sum(d2[6]), endia.sum(d2[7]))

    var d3 = foo_d3(args)
    print(
        "foo_d3 checksum: ", endia.sum(d3[0])
    )  # , endia.sum(d3[1]), endia.sum(d3[2]), endia.sum(d3[3]), endia.sum(d3[4]), endia.sum(d3[5]), endia.sum(d3[6]), endia.sum(d3[7]), endia.sum(d3[8]), endia.sum(d3[9]), endia.sum(d3[10]), endia.sum(d3[11]), endia.sum(d3[12]), endia.sum(d3[13]), endia.sum(d3[14]), endia.sum(d3[15]))
