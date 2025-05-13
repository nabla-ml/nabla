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

import nabla


def foo(args: List[nabla.Array]) -> List[nabla.Array]:
    x = args[0]
    y = args[1]
    return List(x * x, y * y)


fn test_vjp_vjp() raises -> None:
    print("\033[1;33m\nTEST VJP VJP\033[0m")

    # Define inputs
    var x = nabla.arange((2, 3))
    var y = nabla.arange((2, 3))
    var v = nabla.arange((2, 3))
    var w = nabla.arange((2, 3))

    # Step 1: Compute the gradient using VJP with cotangent ones
    def grad_fn(args: List[nabla.Array]) -> List[nabla.Array]:
        _, pullback = nabla.vjp(foo, args)
        return pullback(List(nabla.ones((2, 3)), nabla.ones((2, 3))))

    # Step 2: Compute Hessian-vector product using VJP on grad_fn
    def hvp_fn(args: List[nabla.Array]) -> List[nabla.Array]:
        var primals = args[: len(args) // 2]
        var tangents = args[len(args) // 2 :]
        _, pullback_grad = nabla.vjp(grad_fn, primals)
        return pullback_grad(tangents)

    var res = foo(List(x, y))
    print(res[0])

    print("\nFirst Order VJP (equivalent to jvp result):")
    grad_vjp = grad_fn(List(x, y))
    print(grad_vjp[0])
    print(grad_vjp[1])

    print("\nSecond Order VJP:")
    hvp_result = hvp_fn(List(x, y, v, w))
    print(hvp_result[0])
    print(hvp_result[1])
