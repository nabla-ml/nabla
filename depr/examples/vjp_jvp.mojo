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
    return [x * x, y * y]


def test_vjp_jvp():
    var x = nabla.ndarange((2, 3)) + 2
    var y = nabla.ndarange((2, 3)) + 3
    var v = nabla.ndarange((2, 3))
    var w = nabla.ndarange((2, 3))

    # Step 1: Compute the gradient using JVP.
    def grad_fn(args: List[nabla.Array]) -> List[nabla.Array]:
        _, jvp_result = nabla.jvp(
            foo, args, [nabla.ones((2, 3)), nabla.ones((2, 3))]
        )
        return jvp_result  # This extracts f'(x)

    # Step 2: Compute Hessian-vector product using VJP on grad_fn.
    def hvp_fn(args: List[nabla.Array]) -> List[nabla.Array]:
        var x = args[: len(args) // 2]
        var v = args[len(args) // 2 :]
        _, vjp_fn = nabla.vjp(grad_fn, x)
        return vjp_fn(v)  # VJP on gradient computes H(x)·v

    grad_result = grad_fn([x, y])
    hvp_result = hvp_fn([x, y, v, w])

    print("\nGradient using JVP:")
    print(grad_result[0])
    print(grad_result[1])

    print("\nHessian-vector product using VJP then JVP:")
    print(hvp_result[0])
    print(hvp_result[1])
