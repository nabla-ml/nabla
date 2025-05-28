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


fn test_jacfwd_jacfwd() raises:
    var x = nabla.arange((2, 3)) + 2
    var y = nabla.arange((2, 3)) + 3

    def foo(args: List[nabla.Array]) -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        return [nabla.sin(x) + x**2 + y**2, nabla.sin(x) + x**2 + y**2]

    var jacobian = nabla.jacfwd(foo)
    var jac_res = jacobian([x, y])
    print(nabla.xpr(jacobian)([x, y]))
    print("\nJacobian 1:")
    print(jac_res[0])
    print("\nJacobian 2:")
    print(jac_res[1])
    print("\nJacobian 3:")
    print(jac_res[2])
    print("\nJacobian 4:")
    print(jac_res[3])

    fn jacfwd_foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res = nabla.jacfwd(foo)(args)
        return res

    var hessian = nabla.jacfwd(jacfwd_foo)
    hessian_result = hessian([x, y])
    print(nabla.xpr(hessian)([x, y]))

    print("\nHessian 1:")
    print(hessian_result[0])
    print("\nHessian 2:")
    print(hessian_result[1])
    print("\nHessian 3:")
    print(hessian_result[2])
    print("\nHessian 4:")
    print(hessian_result[3])
    print("\nHessian 5:")
    print(hessian_result[4])
    print("\nHessian 6:")
    print(hessian_result[5])
    print("\nHessian 7:")
    print(hessian_result[6])
    print("\nHessian 8:")
    print(hessian_result[7])
