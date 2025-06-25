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


fn test_jvp_jvp() raises -> None:
    var x = nabla.ndarange((2, 3)) + 2
    var y = nabla.ndarange((2, 3)) + 3

    # Step 1: Compute the gradient using VJP.
    fn grad_fn(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return nabla.jvp(
            foo, args, [nabla.ones_like(args[0]), nabla.ones_like(args[1])]
        )[
            1
        ]  # This extracts f'(x)

    print("First Order JVP:")
    jvp_result = nabla.vmap(grad_fn, [nabla.none, 0])([x, y])
    print(nabla.xpr(nabla.vmap(grad_fn, [nabla.none, 0]))([x, y]))
    print(jvp_result[0])
    # print(jvp_result[1])

    # Step 2: Compute Hessian-vector product using JVP on grad_fn.
    fn hvp_fn_raw(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var primals = args[: len(args) // 2]
        var tangents = args[len(args) // 2 :]
        # return nabla.jvp(grad_fn, primals, tangents)[1]
        return nabla.jvp(
            nabla.vmap(grad_fn, [nabla.none, 0]), primals, tangents
        )[
            1
        ]  # JVP on gradient computes H(x)·v

    var v = nabla.ndarange((2, 3))
    var w = nabla.ndarange((2, 3))

    print("\nSecond Order JVP:")
    hvp_fn = nabla.vmap(hvp_fn_raw)
    hvp_result = hvp_fn([x, y, v, w])
    print(nabla.xpr(hvp_fn)([x, y, v, w]))
    print(hvp_result[0])
    # # print(hvp_result[1])

    # var foo_vmapped = nabla.vmap(foo, [nabla.none, 0])
    # print(nabla.xpr(foo_vmapped)([x,y]))
    # # res = foo_vmapped([x,y])
    # # print(res[0])
