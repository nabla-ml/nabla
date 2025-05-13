# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
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


fn test_jvp() raises:
    var a = nabla.ones((400, 400), DType.float32) / 1000
    var b = nabla.ones((400, 400), DType.float32) / 1000
    var c = nabla.ones((400, 400), DType.float32) / 1000

    var a_tangent = nabla.ones((400, 400), DType.float32) / 1000
    var b_tangent = nabla.ones((400, 400), DType.float32) / 1000
    var c_tangent = nabla.ones((400, 400), DType.float32) / 1000

    fn foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var a = args[0]
        var b = args[1]
        var c = args[2]
        var x = a
        for _ in range(20):
            x = nabla.relu(x @ b + c)
        var z = nabla.sum(nabla.sin(x))
        return List(z, z)

    var avg_time = Float64(0.0)

    var iterations = 4
    var every = 1

    for i in range(1, iterations + 1):
        var start = perf_counter()
        _, tangents = nabla.jvp(
            foo, List(a, b, c), List(a_tangent, b_tangent, c_tangent)
        )

        var res0 = tangents[0].item()
        var res1 = tangents[1].item()
        var end = perf_counter()
        avg_time += end - start

        if i % every == 0:
            print("\nITERATION:", i)
            print("res[0]:", res0)
            print("res[1]:", res1)
            print("TIME:", avg_time / Float64(every))
            avg_time = Float64(0.0)
