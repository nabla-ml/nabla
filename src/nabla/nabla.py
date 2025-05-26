"""
Nabla: A clean, modular deep learning framework built on MAX.

This is the main entry point that provides a clean API while maintaining
backward compatibility with the original monolithic implementation.
"""

# Core exports - The foundation
from .core.array import Array
from .core.execution_context import ThreadSafeExecutionContext
from .core.graph_execution import realize_
from .utils.broadcasting import get_broadcasted_shape

# Operation exports - Clean OOP-based operations
from .ops.binary import add, mul
from .ops.unary import sin, cos, negate
from .ops.linalg import matmul
from .ops.view import transpose, reshape, broadcast_to
from .ops.reduce import sum
from .ops.creation import array, arange, randn

# Set global execution mode
from .ops.base import EAGERMODE

# For backward compatibility, also export the original function names
# These map to the new clean implementations
__all__ = [
    # Core
    "Array",
    "realize_",
    "EAGERMODE",
    "get_broadcasted_shape",
    # Array creation
    "array",
    "arange",
    "randn",
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
    "sum",
    # Creation operations
    "arange",
    "randn",
]

# Maintain the execution context for compatibility
_global_execution_context = ThreadSafeExecutionContext()
