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


def test_eager_mode():
    # EAGER MODE
    def foo1(_args: List[nabla.Array]) -> List[nabla.Array]:
        x = nabla.cast(nabla.sin(_args[0]), DType.float64)
        y = nabla.cast(nabla.sin(_args[1]), DType.float64)
        z = x * y
        if z.load(0) > 0:
            w = nabla.cos(z)
            p = -nabla.sin(z) / y
            return List(w, p)
        else:
            w = nabla.cos(nabla.sin(z))
            p = -nabla.sin(nabla.sin(z))
            return List(w, p)

    ctx = nabla.ExecutionContext()
    arg0 = nabla.ones((1, 3), DType.float32, False, ctx)
    arg1 = nabla.ones((1, 3), DType.float32, False, ctx)

    for iteration in range(1001):
        outputs = foo1(List(arg0, arg1))
        arg0 = outputs[0]
        arg1 = outputs[1]

        if iteration % 1000 == 0:
            print("\nITERATION:", iteration)
            print(arg0)
            print(arg1)


def test_backward_eager_mode():
    # EAGER MODE

    def foo1(_args: List[nabla.Array]) -> List[nabla.Array]:
        x = nabla.cast(nabla.sin(_args[0]), DType.float64)
        y = nabla.cast(nabla.sin(_args[1]), DType.float64)
        z = x * y
        if z.load(0) > 0:
            w = nabla.cos(z)
            p = -nabla.sin(z) / y
            return List(w, p)
        else:
            w = nabla.cos(nabla.sin(z))
            p = -nabla.sin(nabla.sin(z))
            return List(w, p)

    ctx = nabla.ExecutionContext()
    arg0 = nabla.ones((1, 3), DType.float32, True, ctx)
    arg1 = nabla.ones((1, 3), DType.float32, True, ctx)

    for iteration in range(101):
        outputs = foo1(List(arg0, arg1))
        nabla.backward(outputs[0])
        arg0 = arg0 - arg0.grad() * 0.01
        arg1 = arg1 - arg1.grad() * 0.01
        arg0.requires_grad_(True)
        arg1.requires_grad_(True)

        if iteration % 100 == 0:
            print("\nITERATION:", iteration)
            print(arg0)
            print(arg1)
