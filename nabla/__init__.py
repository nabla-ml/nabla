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

"""
Nabla: A clean, modular deep learning framework built on MAX.

This is the main entry point that provides a clean API.
"""

import sys

from max.dtype import DType

# Core exports - The foundation
from .core.array import Array
from .core.execution_context import ThreadSafeExecutionContext
from .core.graph_execution import realize_
from .core.trafos import jvp, vjp, xpr

# Set global execution mode
from .ops.base import EAGERMODE

# Operation exports - Clean OOP-based operations
from .ops.binary import add, mul
from .ops.creation import arange, array, ones, ones_like, randn, zeros, zeros_like
from .ops.linalg import matmul
from .ops.reduce import reduce_sum
from .ops.unary import cos, negate, sin
from .ops.view import broadcast_to, reshape, transpose
from .utils.broadcasting import get_broadcasted_shape

__all__ = [
    # Core
    "Array",
    "realize_",
    "vjp",
    "jvp",
    "EAGERMODE",
    "get_broadcasted_shape",
    # Array creation
    "array",
    "arange",
    "randn",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    # Binary operations
    "add",
    "mul",
    # Unary operations
    "sin",
    "cos",
    "negate",
    # Linear algebra
    "matmul",
    # View operations
    "transpose",
    "reshape",
    "broadcast_to",
    # Reduction operations
    "reduce_sum",
    # Data types
    DType,
]

# Maintain the execution context for compatibility
_global_execution_context = ThreadSafeExecutionContext()

__version__ = "0.1.0"

# For test compatibility - provide a reference to this module
graph_improved = sys.modules[__name__]
