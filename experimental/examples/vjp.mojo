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

import math
from time import perf_counter
import nabla


fn test_vjp() raises:
    fn foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var a = args[0]
        var b = args[1]
        var c = args[2]
        var x = a
        for _ in range(20):
            x = nabla.relu(x @ b + c)
        var z = nabla.sum(nabla.sin(x))
        return [
            z,
        ]

    print("VJP TEST")
    var a = nabla.ones((400, 400), DType.float32) / 1000
    var b = nabla.ones((400, 400), DType.float32) / 1000
    var c = nabla.ones((400, 400), DType.float32) / 1000

    _, foo_vjp = nabla.vjp(foo, [a, b, c])
    foo_vjp_jit = nabla.jit(foo_vjp)

    # loop and measure time
    var avg_time = Float64(0.0)
    var iterations = 4
    var every = 1

    var tangent = nabla.ones((1,), DType.float32)

    for i in range(1, iterations + 1):
        var start = perf_counter()
        var res = foo_vjp_jit(
            [
                tangent,
            ]
        )
        var a_grad = nabla.sum(res[0])
        var b_grad = nabla.sum(res[1])
        var c_grad = nabla.sum(res[2])
        var end = perf_counter()
        avg_time += end - start

        if i % every == 0:
            print("\nITERATION:", i)
            print("a_grad:", a_grad)
            print("b_grad:", b_grad)
            print("c_grad:", c_grad)
            print("TIME:", avg_time / Float64(every))
            avg_time = Float64(0.0)
