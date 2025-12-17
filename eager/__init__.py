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
from .context import (
    defaults,
    default_device,
    default_dtype,
    defaults_like,
)

# Core tensor infrastructure
from .tensor_impl import (
    TensorImpl,
    get_topological_order,
    print_computation_graph,
)

# Compute graph
from .compute_graph import GRAPH, driver_tensor_type, compile_with_sharding

# Main Tensor class
from .tensor import Tensor

# Operation base classes
from .ops import Operation, BinaryOperation

# View operations (for vmap support)
from .view_ops import (
    unsqueeze,
    squeeze,
    swap_axes,
    moveaxis,
    broadcast_to,
    incr_batch_dims,
    decr_batch_dims,
    move_axis_to_batch_dims,
    move_axis_from_batch_dims,
)

# Binary operations
from .binary_ops import (
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

# Creation operations (including random)
from .creation import (
    constant,
    full,
    zeros,
    ones,
    arange,
    uniform,
    gaussian,
    normal,
    CreationOp,
)

# Pytree utilities
from .pytree import (
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


# Sharding infrastructure (core definitions)
from .sharding import (
    DeviceMesh,
    DimSpec,
    ShardingSpec,
    compute_local_shape,
    get_num_shards,
)
from .compute_graph import GRAPH, driver_tensor_type, compile_with_sharding

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
    # Operations
    "Operation",
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
    # View operations
    "unsqueeze",
    "squeeze",
    "swap_axes",
    "moveaxis",
    "broadcast_to",
    "incr_batch_dims",
    "decr_batch_dims",
    "move_axis_to_batch_dims",
    "move_axis_from_batch_dims",
    # Pytree
    "PyTreeDef",
    "broadcast_prefix",
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
    "CreationOp",
    # Sharding
    "DeviceMesh",
    "DimSpec",
    "ShardingSpec",
    "compute_local_shape",
    "get_num_shards",
    "compile_with_sharding",
]
