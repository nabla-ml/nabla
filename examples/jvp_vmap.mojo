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
    return List(nabla.sin(x) + x**2 + y**2, nabla.cos(y) + y * x)


def test_jvp_vmap():
    var x = nabla.arange((2, 3)) + 2
    var y = nabla.arange((2, 3)) + 3
    var v = nabla.arange((3, 2, 3))
    var w = nabla.arange((3, 2, 3))

    # Step 1: Compute the gradient using VJP.
    fn reg_jvp(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var primals = args[: len(args) // 2]
        var tangents = args[len(args) // 2 :]
        return nabla.jvp(foo, primals, tangents)[1]  # This extracts f'(x)

    jvp_vmapped = nabla.vmap(reg_jvp, in_axes=List(nabla.none, nabla.none, 0, 0))
    jvp_result = jvp_vmapped(List(x, y, v, w))

    print("First Order JVP:")
    # print(nabla.xpr(grad_fn)(List(x)))
    print(jvp_result[0])
    print(jvp_result[1])
