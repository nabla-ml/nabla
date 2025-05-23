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


from examples import *


def main():
    # print("\033[1;31m\nANIMATION\033[0m")
    # test_animation()

    # print("\033[1;33m\nCUSTOM OP WITH CUSTOM KERNEL\033[0m")
    # test_custom_op_with_custom_kernel()

    # print("\033[1;94m\nEAGER MODE\033[0m")
    # test_eager_mode()

    # print("\033[1;95m\nBACKWARD EAGER MODE\033[0m")
    # test_backward_eager_mode()

    print("\033[1;32m\nSIMPLE NN\033[0m")
    test_simple_nn()

    # print("\033[1;35m\nVMAP\033[0m")
    # test_vmap()

    # print("\033[1;38m\nVMAP2\033[0m")
    # test_vmap2()

    # print("\033[1;38m\nVMAP3\033[0m")
    # test_vmap3()

    # print("\033[1;38m\nVMAP4\033[0m")
    # test_vmap4()

    # print("\033[1;31m\nMOTREE\033[0m")
    # test_motree()

    # print("\033[1;32m\nMOTREE FUNC\033[0m")
    # test_motree_func()

    # print("\033[1;34m\nVJP\033[0m")
    # test_vjp()

    # print("\033[1;33m\nJVP\033[0m")
    # test_jvp()

    # print("\033[1;94m\nJVP JVP\033[0m")
    # test_jvp_jvp()

    # print("\033[1;95m\nVJP JVP\033[0m")
    # test_vjp_jvp()

    # print("\033[1;96m\nJVP VJP\033[0m")
    # test_jvp_vjp()

    # print("\033[1;96m\nJVP VMAP\033[0m")
    # test_jvp_vmap()

    # print("\033[1;96m\nVJP JACFWD\033[0m")
    # test_vjp_vjp()

    # print("\033[1;33m\nVMAP JVP\033[0m")
    # test_vmap_jvp()

    # print("\033[1;94m\nVMAP VJP\033[0m")
    # test_vmap_vjp()

    # print("\033[1;33m\nVJP VMAP\033[0m")
    # test_vjp_vmap()

    # print("\033[1;33m\nVMAP JACFWD\033[0m")
    # test_vmap_jacfwd()

    # print("\033[1;96m\nJACFWD VMAP\033[0m")
    # test_jacfwd_vmap()

    # print("\033[1;96m\nJACFWD JACFWD\033[0m")
    # test_jacfwd_jacfwd()

    # print("\033[1;33m\nJACREV\033[0m")
    # test_jacrev_jacrev()

    # print("\033[1;33m\nJACREV JACFWD\033[0m")
    # test_jacrev_jacfwd()

    # print("\033[1;33m\nCOMPLEX DERIVATIVES\033[0m")
    # test_jacrev_jacfwd_jacfwd()

    # print("\033[1;33m\nGRAD GRAD GRAD\033[0m")
    # test_grad_grad_grad()
