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


from tests import *


def main():
    test_broadcast0()
    test_broadcast1()
    test_broadcast2()
    test_broadcast3()
    test_broadcast4()
    test_broadcast5()
    test_broadcast6()

    test_reshape0()
    test_reshape1()
    test_reshape2()
    test_reshape3()

    test_squeeze0()
    test_squeeze1()
    test_squeeze2()
    test_squeeze3()

    test_unsqueeze0()
    test_unsqueeze1()
    test_unsqueeze2()
    test_unsqueeze3()

    test_sum0()
    test_sum1()
    test_sum2()
    test_sum3()
    test_sum4()
    test_sum5()
    test_sum6()
    test_sum7()
    test_sum8()
    test_sum9()
    test_sum10()
    test_sum11()
    test_sum12()
    test_sum13()
    test_sum14()

    test_slice0()
    test_slice1()
    test_slice2()

    test_binary_op0()
    test_binary_op1()
    test_binary_op2()
    test_binary_op3()
    test_binary_op4()
    test_binary_op5()
    test_binary_op6()
    test_binary_op7()

    test_broadcast_vjp0()
    test_broadcast_vjp1()
    test_broadcast_vjp2()
    test_broadcast_jacrev0()
