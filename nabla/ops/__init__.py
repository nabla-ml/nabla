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

"""Operations module for Nabla framework."""

# Import all operations for easy access
from .base import EAGERMODE, register_binary_op, register_unary_op
from .binary import add, mul
from .creation import arange, randn
from .linalg import matmul
from .reduce import reduce_sum
from .unary import cast, cos, negate, sin
from .view import broadcast_to, reshape, transpose

__all__ = [
    # Creation operations
    "arange",
    "randn",
    # Unary operations
    "sin",
    "cos",
    "negate",
    "cast",
    # Binary operations
    "add",
    "mul",
    # Linear algebra
    "matmul",
    # View operations
    "transpose",
    "reshape",
    "broadcast_to",
    # Reduction operations
    "reduce_sum",
    # Base utilities
    "register_unary_op",
    "register_binary_op",
    "EAGERMODE",
]
