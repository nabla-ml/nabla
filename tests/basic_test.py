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

import nabla as nb

if __name__ == "__main__":
    n0 = nb.randn((8, 8))
    n1 = nb.randn((4, 8, 8))
    n = nb.reduce_sum(
        nb.reshape(nb.sin(n0 + n1 * n1 + n0), shape=(2, 2, 8, 8)),
        axes=(0, 1, 2),
        keep_dims=False,
    )
    n.realize()
    print(n)
   