# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from .common import (
    defaults, default_device, default_dtype, defaults_like, _in_running_loop,
    tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_structure, PyTreeDef, 
    tensor_leaves, traced, untraced, with_batch_dims,
    pytree  # Re-export module
)
from .tensor import Tensor, TensorImpl
from .graph import (
    ComputeGraph, GRAPH, driver_tensor_type,
    get_operations_topological, get_all_impls_topological, print_trace_graph, apply_to_operations,
    OutputRefs, Trace, trace
)

__all__ = [
    # Tensor
    "Tensor",
    "TensorImpl",
    "OutputRefs",
    # Context
    "defaults",
    "default_device",
    "default_dtype",
    "defaults_like",
    # Pytree
    "tree_map",
    "tree_flatten",
    "tree_unflatten",
    "tree_leaves",
    "tree_structure",
    "PyTreeDef",
    "tensor_leaves",
    "traced",
    "untraced",
    "with_batch_dims",
    # Compute graph
    "ComputeGraph",
    "GRAPH",
    "driver_tensor_type",

    "get_operations_topological",
    "get_all_impls_topological",
    "print_trace_graph",
    "apply_to_operations",
    # TensorImpl utilities
    "Trace",
    "trace",
]
