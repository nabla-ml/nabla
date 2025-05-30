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
"""Tensor utilities for working with tensor-like objects"""

from tensor import (
    DynamicTensor,
    ManagedTensorSlice,
    InputTensor,
    OutputTensor,
    IOSpec,
    Input,
    Output,
    MutableInput,
    StaticTensorSpec,
    _indexing,
    foreach,
)
