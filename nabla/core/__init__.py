"""Core abstractions and runtime for the nabla module."""

from .common import (
    defaults, default_device, default_dtype, defaults_like, _in_running_loop,
    tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_structure, PyTreeDef, 
    tensor_leaves, traced, untraced, with_batch_dims,
    pytree  # Re-export module
)
from .tensor import Tensor, TensorImpl, get_topological_order, print_computation_graph
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
    "get_topological_order",
    "print_computation_graph",
    "Trace",
    "trace",
]
