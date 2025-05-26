"""Operations module for Nabla framework."""

# Import all operations for easy access
from .creation import arange, randn
from .unary import sin, cos, negate, cast
from .binary import add, mul
from .linalg import matmul
from .view import transpose, reshape, broadcast_to
from .reduce import sum
from .base import register_unary_op, register_binary_op, EAGERMODE

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
    "sum",
    # Base utilities
    "register_unary_op",
    "register_binary_op",
    "EAGERMODE",
]
