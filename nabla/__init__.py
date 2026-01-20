# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

# Context managers and defaults
from .core import (
    defaults,
    default_device,
    default_dtype,
    defaults_like,
)

# Core tensor infrastructure
from .core import (
    TensorImpl,
)

# Compute graph
from .core import GRAPH, driver_tensor_type

# Main Tensor class
from .core import Tensor

# Operation base classes
from .ops.base import Operation, BinaryOperation, ReduceOperation, UnaryOperation

# View operations (for vmap support)
from .ops.view import (
    unsqueeze,
    squeeze,
    swap_axes,
    broadcast_to,
    reshape,
    gather,
    scatter,
    concatenate,
    stack,
    moveaxis,
    incr_batch_dims,
    decr_batch_dims,
    move_axis_to_batch_dims,
    move_axis_from_batch_dims,
    unsqueeze_physical,
    squeeze_physical,
    broadcast_to_physical,
)
from .ops.reduction import reduce_sum_physical, mean_physical

# Binary operations
from .ops.binary import (
    add,
    mul,
    sub, 
    div,
    matmul,
    AddOp,
    MulOp,
    SubOp,
    DivOp,
    MatmulOp,
)

# Unary operations
from .ops.unary import (
    relu,
    sigmoid,
    tanh,
    exp,
    neg,
    abs,
    softmax,
    ReluOp,
    SigmoidOp,
    TanhOp,
    ExpOp,
    NegOp,
    AbsOp,
    AbsOp,
)

# Comparison operations
from .ops.comparison import (
    equal,
    not_equal,
    greater,
    greater_equal,
    less,
    less_equal,
    EqualOp,
    NotEqualOp,
    GreaterOp,
    GreaterEqualOp,
    LessOp,
    LessEqualOp,
)

# Creation operations (including random)
from .ops.creation import (
    constant,
    full,
    zeros,
    ones,
    arange,
    uniform,
    gaussian,
    normal,
)

# Pytree utilities
from .core import (
    PyTreeDef,
    tensor_leaves,
    traced,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
    untraced,
    with_batch_dims,
)

# Function transforms
from .transforms.vmap import vmap

from .transforms.shard_map import shard_map
from .transforms.compile import compile, CompiledFunction, CompilationStats

# Reduction operations
from .ops.reduction import (
    reduce_sum, mean, reduce_max, reduce_min,
    ReduceSumOp, MeanOp, ReduceMaxOp, ReduceMinOp,
)

# Sharding operations
from .ops.communication import shard, all_gather, all_reduce

# Control Flow
from .ops.control_flow import (
    where,
    cond,
    while_loop,
    scan,
    WhereOp,
    CondOp,
    WhileLoopOp,
    ScanOp,
)

# Multi-output operations
from .ops.multi_output import (
    split,
    chunk,
    unbind,
    minmax,
    SplitOp,
    ChunkOp,
    UnbindOp,
    MinMaxOp,
)

# Sharding infrastructure (core definitions)
from .core.sharding.spec import (
    DeviceMesh,
    DimSpec,
    ShardingSpec,
    compute_local_shape,
    get_num_shards,
    P,
    PartitionSpec,
)

__all__ = [
    # Context
    "defaults",
    "default_device", 
    "default_dtype",
    "defaults_like",
    # Core
    "Tensor",
    "TensorImpl",
    "GRAPH",
    "driver_tensor_type",
    # Operations (base classes)
    "Operation",
    "BinaryOperation",
    "ReduceOperation",
    "UnaryOperation",
    # Binary operations
    "add",
    "mul",
    "sub",
    "div",
    "matmul",
    "AddOp",
    "MulOp",
    "SubOp",
    "DivOp",
    "MatmulOp",
    # Reduction operations
    "ReduceSumOp",
    "MeanOp",
    "ReduceMaxOp",
    "ReduceMaxOp",
    "ReduceMinOp",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "mean",
    # Sharding operations
    "shard",
    "all_gather",
    "all_reduce",
    # Control Flow
    "where",
    "cond",
    "while_loop",
    "scan",
    "WhereOp",
    "CondOp",
    "WhileLoopOp",
    "ScanOp",
    # Multi-output operations
    "split",
    "chunk",
    "unbind",
    "minmax",
    "SplitOp",
    "ChunkOp",
    "UnbindOp",
    "MinMaxOp",
    # Comparison operations
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "EqualOp",
    "NotEqualOp",
    "GreaterOp",
    "GreaterEqualOp",
    "LessOp",
    "LessEqualOp",
    # Unary operations
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "neg",
    "abs",
    "softmax",
    "ReluOp",
    "SigmoidOp",
    "TanhOp",
    "ExpOp",
    "NegOp",
    "AbsOp",
    "AbsOp",
    # View operations
    "unsqueeze",
    "squeeze",
    "swap_axes",
    "moveaxis",
    "broadcast_to",
    "reshape",
    "gather",
    "scatter", 
    "concatenate",
    "stack",
    "incr_batch_dims",
    "decr_batch_dims",
    "move_axis_to_batch_dims",
    "move_axis_from_batch_dims",
    "unsqueeze_physical",
    "squeeze_physical",
    "broadcast_to_physical",
    "reduce_sum_physical",
    "mean_physical",
    # Transforms
    "vmap",
    "shard_map",
    "compile",
    "CompiledFunction",
    "CompilationStats",
    # Pytree
    "PyTreeDef",
    "tensor_leaves",
    "traced",
    "tree_flatten",
    "tree_leaves",
    "tree_map",
    "tree_structure",
    "tree_unflatten",
    "untraced",
    "with_batch_dims",
    # Creation (including random)
    "constant",
    "full",
    "zeros",
    "ones",
    "arange",
    "uniform",
    "gaussian",
    "normal",
    # Sharding
    "DeviceMesh",
    "DimSpec",
    "ShardingSpec",
    "P",
    "PartitionSpec",
    "compute_local_shape",
    "get_num_shards",
    # Debug
    "xpr",
    "capture_trace",
]
