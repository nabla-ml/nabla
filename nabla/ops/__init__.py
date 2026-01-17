# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #


# Base classes
from .operation import Operation, BinaryOperation, UnaryOperation, ReduceOperation, LogicalShapeOperation, LogicalAxisOperation, ensure_tensor

# Binary operations
from .binary import add, mul, sub, div, matmul, AddOp, MulOp, SubOp, DivOp, MatmulOp

# Unary operations
from .unary import relu, sigmoid, tanh, exp, neg, softmax, ReluOp, SigmoidOp, TanhOp, ExpOp, NegOp, SoftmaxOp

# Creation operations
from .creation import constant, full, zeros, ones, arange, uniform, gaussian, normal
from .creation import ConstantOp, FullOp, ZerosOp, OnesOp, ArangeOp, UniformOp, GaussianOp

# Comparison operations
from .comparison import equal, not_equal, greater, greater_equal, less, less_equal
from .comparison import EqualOp, NotEqualOp, GreaterOp, GreaterEqualOp, LessOp, LessEqualOp

# Control Flow operations
from .control_flow import where, cond, while_loop, scan
from .control_flow import WhereOp, CondOp, WhileLoopOp

# Reduction operations
from .reduction import reduce_sum, mean, ReduceSumOp, MeanOp

# View operations (logical)
from .view import unsqueeze, squeeze, swap_axes, broadcast_to, reshape, slice_tensor, gather, scatter, stack, concatenate
from .view import UnsqueezeOp, SqueezeOp, SwapAxesOp, BroadcastToOp, ReshapeOp, SliceTensorOp, GatherOp, ScatterOp, ConcatenateOp


# Multi-output operations
from .multi_output import split, chunk, unbind, minmax
from .multi_output import SplitOp, ChunkOp, UnbindOp, MinMaxOp

# Communication operations (sharding)
from .communication import shard, all_gather, all_reduce
from .communication import ShardOp, AllGatherOp, AllReduceOp, ReduceScatterOp

# Note: _physical is internal only, not exported
# Physical operations can be imported via: from nabla.ops._physical import ...

from .custom_op import call_custom_kernel

__all__ = [
    # Base classes
    "Operation",
    "BinaryOperation",
    "UnaryOperation",
    "ReduceOperation",
    "LogicalShapeOperation",
    "LogicalAxisOperation",
    "ensure_tensor",
    # Binary
    "add", "mul", "sub", "div", "matmul",
    "AddOp", "MulOp", "SubOp", "DivOp", "MatmulOp",
    # Unary
    "relu", "sigmoid", "tanh", "exp", "neg", "softmax",
    "ReluOp", "SigmoidOp", "TanhOp", "ExpOp", "NegOp", "SoftmaxOp",
    # Creation
    "constant", "full", "zeros", "ones", "arange", "uniform", "gaussian", "normal",
    "ConstantOp", "FullOp", "ZerosOp", "OnesOp", "ArangeOp", "UniformOp", "GaussianOp",
    # Comparison
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "EqualOp", "NotEqualOp", "GreaterOp", "GreaterEqualOp", "LessOp", "LessEqualOp",
    # Control Flow
    "where", "cond", "while_loop", "scan",
    "WhereOp", "CondOp", "WhileLoopOp",
    # Reduction
    "reduce_sum", "mean",
    "ReduceSumOp", "MeanOp",
    # View (logical)
    "unsqueeze", "squeeze", "swap_axes", "broadcast_to", "reshape", "slice_tensor", "gather", "scatter", "stack", "concatenate",
    "UnsqueezeOp", "SqueezeOp", "SwapAxesOp", "BroadcastToOp", "ReshapeOp", "SliceTensorOp", "GatherOp", "ScatterOp", "ConcatenateOp",
    # Multi-output
    "split", "chunk", "unbind", "minmax",
    "SplitOp", "ChunkOp", "UnbindOp", "MinMaxOp",
    # Communication (sharding)
    "shard", "all_gather", "all_reduce",
    "ShardOp", "AllGatherOp", "AllReduceOp", "ReduceScatterOp",
    "call_custom_kernel",
]

