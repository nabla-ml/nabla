"""SPMD execution helpers for sharded tensors.

Core utilities for detecting, slicing, aligning, and creating sharded tensors.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from ..core.tensor import Tensor
    from ..sharding.spec import ShardingSpec
    from ..sharding.mesh import DeviceMesh

# ============================================================================
# Detection & Extraction
# ============================================================================


def get_mesh_from_args(args: tuple) -> Optional["DeviceMesh"]:
    """Extract DeviceMesh from first tensor with sharding spec."""
    from ..core.tensor import Tensor
    from ..core import pytree
    for a in pytree.tree_leaves(args):
        if isinstance(a, Tensor) and a._impl.sharding:
            return a._impl.sharding.mesh
    return None


def ensure_specs(args: tuple, mesh: Optional["DeviceMesh"]) -> tuple:
    """Ensure all tensors have explicit sharding specs.
    
    For unsharded tensors, assigns a replicated spec (all dims have empty axes).
    This allows the unified SPMD dispatch to treat all tensors uniformly.
    """
    if mesh is None:
        return args
    
    from ..core.tensor import Tensor
    from ..core import pytree
    from ..sharding.spec import ShardingSpec, DimSpec
    
    def assign_spec(x):
        if not isinstance(x, Tensor):
            return x
        if x._impl.sharding is not None:
            return x
        # Use PHYSICAL rank (logical + batch_dims) for sharding spec
        # Sharding specs must cover ALL physical dimensions
        physical_rank = len(x.shape) + x._impl.batch_dims
        x._impl.sharding = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(physical_rank)])
        return x
    
    return pytree.tree_map(assign_spec, args)


def reshard_inputs(
    args: tuple, 
    required_specs: "List[Optional[ShardingSpec]]", 
    mesh: Optional["DeviceMesh"]
) -> tuple:
    """Pre-operation resharding: align inputs to their required specs.
    
    For each tensor input, compares its current sharding to the required sharding
    from propagation. If they differ, inserts communication ops.
    """
    if mesh is None or not required_specs:
        return args
    
    from ..core.tensor import Tensor
    from ..core import pytree
    
    leaves = [a for a in pytree.tree_leaves(args) if isinstance(a, Tensor)]
    if len(leaves) != len(required_specs):
        return args
    
    tensor_to_spec = {id(t): spec for t, spec in zip(leaves, required_specs)}
    
    def reshard_if_needed(x):
        if not isinstance(x, Tensor):
            return x
        required = tensor_to_spec.get(id(x))
        if required is None:
            return x
        
        current = x._impl.sharding
        
        # If input is replicated/unsharded but required spec has sharded axes,
        # we need to actually shard it to match
        if current is None or current.is_fully_replicated():
            if required is not None and not required.is_fully_replicated():
                # Shard the unsharded tensor to match the required spec
                from ..ops.communication import shard as shard_fn
                return shard_fn(x, mesh, required.dim_specs)
            return x
            
        if not needs_reshard(current, required):
            return x
        return reshard_tensor(x, current, required, mesh)
    
    return pytree.tree_map(reshard_if_needed, args)


def infer_output_sharding(
    op: Any,
    args: tuple,
    mesh: "DeviceMesh",
    kwargs: dict = None,
) -> "Tuple[Optional[ShardingSpec], List[Optional[ShardingSpec]], bool]":
    """Infer output AND per-input shardings using factor-based propagation.
    
    Uses the operation's sharding_rule() to get factor mappings, then calls
    propagate_sharding() to compute shardings. Input specs are updated in-place
    by propagation and returned for per-input slicing.
    
    Returns:
        Tuple of (output_sharding, input_shardings, needs_allreduce)
    """
    if mesh is None:
        return None, [], False
        
    # Hook for Ops that define explicit sharding logic (bypassing propagation)
    if hasattr(op, "infer_sharding_spec"):
        return op.infer_sharding_spec(args, mesh, kwargs)

    from ..core.tensor import Tensor
    from ..core import pytree
    from ..sharding.spec import ShardingSpec, DimSpec
    from ..sharding.propagation import propagate_sharding
    
    # Collect input specs and shapes
    leaves = [a for a in pytree.tree_leaves(args) if isinstance(a, Tensor)]
    if not leaves:
        return None, [], False
    
    def dim_to_int(d):
        """Safely convert Dim object to int."""
        try:
            return int(d)
        except (TypeError, ValueError):
            # Symbolic dim - use a placeholder size (shouldn't happen in practice)
            return 1
    
    input_specs = []
    input_shapes = []
    for t in leaves:
        spec = t._impl.sharding
        
        # Determine global physical shape
        # cached_shape is Global Physical if sharded, or local if unsharded (but local=global)
        # We need physical shape (rank) to create correct default spec
        shape = t.global_shape
        if shape is None and (t._impl.sharding is None or t._impl.sharding.is_fully_replicated()):
            shape = t._impl.physical_shape
            
        if shape is None:
             # Fallback: reconstruct physical shape from logical + batch_shape
             batch_dims = t._impl.batch_dims
             if batch_dims > 0 and t._impl.batch_shape is not None:
                  # Prepend batch dims to logical shape
                  batch_ints = tuple(dim_to_int(d) for d in t._impl.batch_shape)
                  logical_ints = tuple(dim_to_int(d) for d in t.shape)
                  phys_shape_tuple = batch_ints + logical_ints
             else:
                  # No batch dims or batch_shape unavailable - use logical as physical
                  phys_shape_tuple = tuple(dim_to_int(d) for d in t.shape)
        else:
            phys_shape_tuple = tuple(dim_to_int(d) for d in shape)
        
        if spec is None:
            # Create replicated spec for unsharded inputs (OPEN so they can receive sharding)
            # Must use PHYSICAL rank
            rank = len(phys_shape_tuple)
            spec = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
        input_specs.append(spec.clone())
        input_shapes.append(phys_shape_tuple)
    
    if not any(spec.dim_specs and any(d.axes for d in spec.dim_specs) for spec in input_specs):
        return None, input_specs, False  # All inputs replicated
    
    # Use rank inference instead of full shape inference
    try:
        output_rank = op.infer_output_rank(input_shapes, **(kwargs or {}))
    except (NotImplementedError, AttributeError):
        output_rank = len(input_shapes[0]) if input_shapes else 0

    # Get sharding rule from operation
    try:
        # We pass output_shapes=None to indicate we don't have full shapes
        # The template instantiation will rely on input shapes for factor sizing
        rule = op.sharding_rule(input_shapes, None, **(kwargs or {}))
    except (NotImplementedError, AttributeError):
        # Fallback to elementwise-like behavior: inherit from first sharded
        for spec in input_specs:
            if any(d.axes for d in spec.dim_specs):
                return spec, input_specs, False
        return None, input_specs, False
    
    # Create empty output spec based on rank
    output_spec = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(output_rank)])
    
    # Run factor-based propagation - updates input_specs and output_spec in-place
    propagate_sharding(rule, input_specs, [output_spec])
    
    # Check if any contracting factors are sharded (need AllReduce for partial results)
    needs_allreduce = _check_contracting_factors_sharded(rule, input_specs)
    
    return output_spec, input_specs, needs_allreduce


def _check_contracting_factors_sharded(
    rule: "OpShardingRule",
    input_specs: "List[ShardingSpec]",
) -> "Set[str]":
    """Check if any contracting factor has sharding (requires AllReduce).
    
    A contracting factor appears in inputs but not outputs (e.g., k in matmul).
    If such a factor is sharded, the operation produces partial results.
    
    Returns:
        Set of mesh axis names that require AllReduce.
    """
    reduce_axes = set()
    contracting_factors = rule.get_contracting_factors()
    if not contracting_factors:
        return reduce_axes
    
    # For each contracting factor, check if it's sharded in any input
    for factor in contracting_factors:
        for t_idx, mapping in enumerate(rule.input_mappings):
            for dim_idx, factors in mapping.items():
                if factor in factors and t_idx < len(input_specs):
                    spec = input_specs[t_idx]
                    if dim_idx < len(spec.dim_specs):
                        # Add all axes used for this contracting dimension
                        reduce_axes.update(spec.dim_specs[dim_idx].axes)
    return reduce_axes








# ============================================================================
# Shape & Slice Computation
# ============================================================================

def compute_global_shape(local_shape: tuple, sharding: "ShardingSpec") -> tuple:
    """Multiply sharded dims by shard count to get global shape."""
    if not sharding or not local_shape:
        return local_shape
    result = [int(d) for d in local_shape]  # Convert Dim to int
    for i, spec in enumerate(sharding.dim_specs[:len(result)]):
        if spec.axes:
            result[i] *= spec.get_total_shards(sharding.mesh)
    return tuple(result)


def create_replicated_spec(mesh: "DeviceMesh", rank: int) -> "ShardingSpec":
    """Create a fully replicated sharding spec.
    
    Args:
        mesh: Device mesh
        rank: Tensor rank (number of dimensions)
        
    Returns:
        ShardingSpec with all dimensions replicated (empty axes)
    """
    from .spec import ShardingSpec, DimSpec
    return ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])


from .spec import needs_reshard



# ============================================================================
# SPMD Argument Extraction
# ============================================================================

def ensure_shard_values(tensor: "Tensor") -> list:
    """Ensure tensor has symbolic values, hydrating from storage if needed.
    
    If a tensor is realized (has _storages but cleared _values), this creates
    new graph inputs for the storages and populates _values.
    """
    if tensor._impl._values:
        return tensor._impl._values
        
    if tensor._impl._storages and len(tensor._impl._storages) > 0:
        from ..core.compute_graph import GRAPH
        from ..core.tensor import Tensor
        
        # We must be in a graph context to add inputs
        try:
            # Check if we are in a graph context
            _ = GRAPH.graph
            
            with GRAPH.graph:
                new_values = []
                for s in tensor._impl._storages:
                    # Wrap storage as Tensor to adapt to graph input
                    t = Tensor(storage=s)
                    new_values.append(t.__tensorvalue__())
                tensor._impl._values = new_values
                return new_values
        except LookupError:
            # Not in a graph context - return empty or handle gracefully?
            # Operations requiring values will fail anyway.
            pass
            
    return []



def get_shard_args(args: tuple, shard_idx: int, 
                   per_input_shardings: "List[Optional[ShardingSpec]]",
                   g: Any, Tensor: type, pytree: Any) -> tuple:
    """Get per-shard TensorValues, slicing each input according to its OWN sharding.
    
    Args:
        args: Input arguments (may contain Tensors)
        shard_idx: Index of the shard to extract
        per_input_shardings: List of ShardingSpecs, one per input tensor
        g: graph module
        Tensor: Tensor class
        pytree: pytree module
    """
    # Use list for closure mutation
    input_idx = [0]
    
    def extract(x):
        if not isinstance(x, Tensor):
            return x
        
        # Get THIS input's sharding
        this_sharding = None
        if per_input_shardings and input_idx[0] < len(per_input_shardings):
            this_sharding = per_input_shardings[input_idx[0]]
        input_idx[0] += 1
        
        if x._impl.is_sharded:
            vals = x._impl._values
            
            if len(vals) > 1 and shard_idx < len(vals):
                # Distributed/Simulated: Return specific shard
                return vals[shard_idx]
            elif len(vals) == 1:
                return x.__tensorvalue__()
            else:
                return x.__tensorvalue__()
                
        # Unsharded case
        return x.__tensorvalue__()
    
    return pytree.tree_map(extract, args)



# ============================================================================
# Resharding
# ============================================================================

def reshard_tensor(tensor: "Tensor", from_spec: Optional["ShardingSpec"],
                   to_spec: Optional["ShardingSpec"], mesh: "DeviceMesh") -> "Tensor":
    """Reshard tensor from one sharding spec to another."""
    from ..ops.communication import all_gather, shard as shard_op
    from ..sharding.spec import DimSpec
    
    if mesh is None:
        return tensor
        
    if not from_spec:
        from_spec = create_replicated_spec(mesh, len(tensor.shape))
        
    if not needs_reshard(from_spec, to_spec):
        return tensor

    result = tensor
    if from_spec:
        for dim, dim_spec in enumerate(from_spec.dim_specs):
            if dim_spec.axes:
                 result = all_gather(result, axis=dim)
    
    result = shard_op(result, mesh, to_spec.dim_specs)
    
    return result





# ============================================================================
# Output Creation
# ============================================================================

def create_sharded_output(results: List[Any], sharding: Optional["ShardingSpec"],
                          traced: bool, batch_dims: int,
                          mesh: Optional["DeviceMesh"] = None) -> "Tensor":
    """Build sharded Tensor from per-shard TensorValues."""
    from ..core.tensor import Tensor
    from ..core.tensor_impl import TensorImpl
    from max import graph as g
    
    if not results:
        raise ValueError("Empty shard results")
    
    first = results[0]
    if not isinstance(first, (g.TensorValue, g.BufferValue)):
        return first
    
    # Ensure we have a sharding spec when we have multiple values
    # (replicated tensors still need a spec with empty axes for each dim)
    if sharding is None and len(results) > 1 and mesh is not None:
        from .spec import ShardingSpec, DimSpec
        rank = len(first.type.shape)
        # Implicitly replicated tensors should be Open to allow modification during propagation
        sharding = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
    
    impl = TensorImpl(values=results, traced=traced, batch_dims=batch_dims, sharding=sharding)
    
    # Cache metadata - compute GLOBAL shape from local + sharding for sharded outputs
    local_shape = tuple(first.type.shape)
    if sharding and not sharding.is_fully_replicated():
        global_shape = compute_global_shape(local_shape, sharding)
        impl.cached_shape = g.Shape(global_shape)  # Use cached_shape attribute
        impl.cached_dtype = first.type.dtype
    else:
        impl.cache_metadata(first)
    
    return Tensor(impl=impl)
