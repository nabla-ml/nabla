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


fn test_jvp_jvp() raises -> None:
    var x = endia.arange((2, 3)) + 2
    var y = endia.arange((2, 3)) + 3

    # Step 1: Compute the gradient using VJP.
    fn grad_fn(args: List[endia.Array]) raises -> List[endia.Array]:
        return endia.jvp(
            foo, args, [endia.ones_like(args[0]), endia.ones_like(args[1])]
        )[
            1
        ]  # This extracts f'(x)

    print("First Order JVP:")
    jvp_result = endia.vmap(grad_fn, [endia.none, 0])([x, y])
    print(endia.xpr(endia.vmap(grad_fn, [endia.none, 0]))([x, y]))
    print(jvp_result[0])
    # print(jvp_result[1])

    # Step 2: Compute Hessian-vector product using JVP on grad_fn.
    fn hvp_fn_raw(args: List[endia.Array]) raises -> List[endia.Array]:
        var primals = args[: len(args) // 2]
        var tangents = args[len(args) // 2 :]
        # return endia.jvp(grad_fn, primals, tangents)[1]
        return endia.jvp(
            endia.vmap(grad_fn, [endia.none, 0]), primals, tangents
        )[
            1
        ]  # JVP on gradient computes H(x)·v

    var v = endia.arange((2, 3))
    var w = endia.arange((2, 3))

    print("\nSecond Order JVP:")
    hvp_fn = endia.vmap(hvp_fn_raw)
    hvp_result = hvp_fn([x, y, v, w])
    print(endia.xpr(hvp_fn)([x, y, v, w]))
    print(hvp_result[0])
    # # print(hvp_result[1])

    # var foo_vmapped = endia.vmap(foo, [endia.none, 0])
    # print(endia.xpr(foo_vmapped)([x,y]))
    # # res = foo_vmapped([x,y])
    # # print(res[0])
