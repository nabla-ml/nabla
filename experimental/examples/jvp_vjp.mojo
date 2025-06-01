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


def test_jvp_vjp():
    var x = nabla.arange((2, 3)) + 2
    var y = nabla.arange((2, 3)) + 3
    var v = nabla.arange((2, 3))
    var w = nabla.arange((2, 3))

    # Step 1: Compute the gradient using VJP.
    def grad_fn(args: List[nabla.Array]) -> List[nabla.Array]:
        _, vjp_fn = nabla.vjp(foo, args)
        return vjp_fn([nabla.ones((2, 3)), nabla.ones((2, 3))])

    # Step 2: Compute Hessian-vector product using JVP on grad_fn.
    def hvp_fn(args: List[nabla.Array]) -> List[nabla.Array]:
        var primals = args[: len(args) // 2]
        var tangents = args[len(args) // 2 :]
        return nabla.jvp(grad_fn, primals, tangents)[
            1
        ]  # JVP on gradient computes H(x)·v

    grad_result = grad_fn([x, y])
    hvp_result = hvp_fn([x, y, v, w])

    print("\nGradient using VJP:")
    # print(nabla.xpr(grad_fn)([x,]))
    print(grad_result[0])
    print(grad_result[1])

    print("\nHessian-vector product using VJP + JVP:")
    # print(nabla.xpr(hvp_fn)([x, v]))
    print(hvp_result[0])
    print(hvp_result[1])
