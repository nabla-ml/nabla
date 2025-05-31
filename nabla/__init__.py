# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
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
from .core.trafos import jit, jvp, vjp, vmap, xpr

# Operation exports - Clean OOP-based operations
from .ops.binary import add, div, greater_equal, mul, power, sub
from .ops.creation import arange, array, ones, ones_like, randn, zeros, zeros_like
from .ops.linalg import matmul
from .ops.reduce import reduce_sum
from .ops.unary import (
    cos,
    decr_batch_dim_ctr,
    incr_batch_dim_ctr,
    log,
    negate,
    relu,
    sin,
)
from .ops.view import (
    broadcast_batch_dims,
    broadcast_to,
    reshape,
    squeeze,
    transpose,
    unsqueeze,
)
from .utils.broadcasting import get_broadcasted_shape
from .utils.formatting import format_dtype, format_shape_and_dtype
from .utils.max_interop import accelerator, cpu, device

__all__ = [
    "Array",
    "realize_",
    "vjp",
    "jvp",
    "vmap",
    "xpr",
    "jit",
    "get_broadcasted_shape",
    "device",
    "cpu",
    "accelerator",
    "format_dtype",
    "format_shape_and_dtype",
    "array",
    "arange",
    "randn",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "add",
    "mul",
    "sub",
    "div",
    "greater_equal",
    "power",
    "sin",
    "cos",
    "negate",
    "incr_batch_dim_ctr",
    "decr_batch_dim_ctr",
    "relu",
    "log",
    "matmul",
    "transpose",
    "reshape",
    "broadcast_to",
    "squeeze",
    "unsqueeze",
    "broadcast_batch_dims",
    "reduce_sum",
    DType,
]

# Maintain the execution context for compatibility
_global_execution_context = ThreadSafeExecutionContext()

__version__ = "0.1.0"

# For test compatibility - provide a reference to this module
graph_improved = sys.modules[__name__]
