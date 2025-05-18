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
    return [x * x, y * y]


def test_vjp_vmap():
    var x = nabla.arange((3, 2, 3)) + 2
    var y = nabla.arange((3, 2, 3)) + 3

    # Step 1: Compute the gradient using VJP.
    def vjp_vmapped(args: List[nabla.Array]) -> List[nabla.Array]:
        _, vjp_fn = nabla.vjp(nabla.vmap(foo), args)
        return vjp_fn([nabla.ones((3, 2, 3)), nabla.ones((3, 2, 3))])

    vjp_result = vjp_vmapped([x, y])

    print("First Order VJP:")
    # print(nabla.xpr(grad_fn)([x,]))
    print(vjp_result[0])
    print(vjp_result[1])
