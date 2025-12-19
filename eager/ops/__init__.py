"""Operations for tensor manipulation."""

# Base classes
from .operation import Operation, BinaryOperation, UnaryOperation, ReduceOperation, LogicalShapeOperation, LogicalAxisOperation

# Binary operations
from .binary import add, mul, sub, div, matmul, AddOp, MulOp, SubOp, DivOp, MatmulOp

# Unary operations
from .unary import relu, sigmoid, tanh, exp, neg, ReluOp, SigmoidOp, TanhOp, ExpOp, NegOp

# Creation operations
from .creation import constant, full, zeros, ones, arange, uniform, gaussian, normal
from .creation import ConstantOp, FullOp, ZerosOp, OnesOp, ArangeOp, UniformOp, GaussianOp

# Reduction operations
from .reduction import reduce_sum, mean, ReduceSumOp, MeanOp

# View operations (logical)
from .view import unsqueeze, squeeze, swap_axes, broadcast_to, reshape
from .view import UnsqueezeOp, SqueezeOp, SwapAxesOp, BroadcastToOp, ReshapeOp

# Multi-output operations
from .multi_output import split, chunk, unbind, minmax
from .multi_output import SplitOp, ChunkOp, UnbindOp, MinMaxOp

# Note: _physical is internal only, not exported
# Physical operations can be imported via: from eager.ops._physical import ...

__all__ = [
    # Base classes
    "Operation",
    "BinaryOperation",
    "UnaryOperation",
    "ReduceOperation",
    "LogicalShapeOperation",
    "LogicalAxisOperation",
    # Binary
    "add", "mul", "sub", "div", "matmul",
    "AddOp", "MulOp", "SubOp", "DivOp", "MatmulOp",
    # Unary
    "relu", "sigmoid", "tanh", "exp", "neg",
    "ReluOp", "SigmoidOp", "TanhOp", "ExpOp", "NegOp",
    # Creation
    "constant", "full", "zeros", "ones", "arange", "uniform", "gaussian", "normal",
    "ConstantOp", "FullOp", "ZerosOp", "OnesOp", "ArangeOp", "UniformOp", "GaussianOp",
    # Reduction
    "reduce_sum", "mean",
    "ReduceSumOp", "MeanOp",
    # View (logical)
    "unsqueeze", "squeeze", "swap_axes", "broadcast_to", "reshape",
    "UnsqueezeOp", "SqueezeOp", "SwapAxesOp", "BroadcastToOp", "ReshapeOp",
    # Multi-output
    "split", "chunk", "unbind", "minmax",
    "SplitOp", "ChunkOp", "UnbindOp", "MinMaxOp",
]
