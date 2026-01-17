# ===----------------------------------------------------------------------=== #
# Nabla 2026
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


"""Experimental eager execution APIs for the MAX platform."""

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
    get_topological_order,
    print_computation_graph,
)

# Compute graph
from .core import GRAPH, driver_tensor_type

# Main Tensor class
from .core import Tensor

# Operation base classes
from .ops.operation import Operation, BinaryOperation, ReduceOperation, UnaryOperation

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
)

from .ops._physical import (
    moveaxis,
    incr_batch_dims,
    decr_batch_dims,
    move_axis_to_batch_dims,
    move_axis_from_batch_dims,
    unsqueeze_physical,
    squeeze_physical,
    broadcast_to_physical,
    reduce_sum_physical,
    mean_physical,
)

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
    ReluOp,
    SigmoidOp,
    TanhOp,
    ExpOp,
    NegOp,
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
from .ops.reduction import reduce_sum, mean, ReduceSumOp, MeanOp

# Sharding operations
from .ops.communication import shard, all_gather, all_reduce

# Sharding infrastructure (core definitions)
from .sharding.spec import (
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
    "get_topological_order",
    "print_computation_graph",
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
    "reduce_sum",
    "mean",
    "ReduceSumOp",
    "MeanOp",
    # Sharding operations
    "shard",
    "all_gather",
    "all_reduce",
    # Unary operations
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "neg",
    "ReluOp",
    "SigmoidOp",
    "TanhOp",
    "ExpOp",
    "NegOp",
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
