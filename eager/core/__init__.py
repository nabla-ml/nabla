"""Core abstractions and runtime for the eager module."""

from .tensor import Tensor
from .tensor_impl import TensorImpl, get_topological_order, print_computation_graph
from .tracing import OutputRefs
from .context import defaults, default_device, default_dtype, defaults_like, _in_running_loop
from .pytree import tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_structure, PyTreeDef, tensor_leaves, traced, untraced, with_batch_dims
from .compute_graph import ComputeGraph, GRAPH, driver_tensor_type, compile_with_sharding
from .graph_utils import get_operations_topological, get_all_impls_topological, print_trace_graph, apply_to_operations

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
    "compile_with_sharding",
    "get_operations_topological",
    "get_all_impls_topological",
    "print_trace_graph",
    "apply_to_operations",
    # TensorImpl utilities
    "get_topological_order",
    "print_computation_graph",
]
