# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import nabla


def test_shape():
    shape0 = (2, 3)
    shape1 = (2, 2, 3)
    res_shape = nabla.get_broadcasted_shape(shape0, shape1)
    print(res_shape)


if __name__ == "__main__":
    test_shape()
