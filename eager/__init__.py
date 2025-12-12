# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
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
from .compute_graph import GRAPH, driver_tensor_type

# Main Tensor class
from .tensor import Tensor

# Operation base class
from .ops import Operation

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

# Pytree utilities
from .pytree import (
    PyTreeDef,
    broadcast_prefix,
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

# Random generation
from . import random

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
    # Random
    "random",
]
