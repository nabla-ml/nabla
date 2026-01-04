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

def has_sharded_inputs(args: tuple) -> bool:
    """True if any tensor in args is sharded."""
    from ..core.tensor import Tensor
    from ..core import pytree
    return any(isinstance(a, Tensor) and a._impl.is_sharded for a in pytree.tree_leaves(args))


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
        rank = len(x.shape)
        x._impl.sharding = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
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
        
        # Optimization: If input is replicated/unsharded, do NOT insert a Reshard op.
        # get_shard_args will handle slicing the replicated tensor directly.
        if current is None or current.is_fully_replicated():
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
    
    input_specs = []
    input_shapes = []
    for t in leaves:
        spec = t._impl.sharding
        if spec is None:
            # Create replicated spec for unsharded inputs (OPEN so they can receive sharding)
            rank = len(t.shape)
            spec = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
        input_specs.append(spec.clone())
        input_shapes.append(tuple(int(d) for d in t.shape))
    
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


def needs_reshard(from_spec: Optional["ShardingSpec"], to_spec: Optional["ShardingSpec"]) -> bool:
    """Check if specs differ requiring resharding."""
    if (from_spec is None) != (to_spec is None):
        return True
    if from_spec is None:
        return False
    if len(from_spec.dim_specs) != len(to_spec.dim_specs):
        return True
    return any(f.axes != t.axes for f, t in zip(from_spec.dim_specs, to_spec.dim_specs))







def slice_for_shard(tensor_val: Any, shape: tuple, sharding: "ShardingSpec", shard_idx: int) -> Any:
    """Slice replicated tensor to extract portion for given shard."""
    slices = []
    for dim, dim_len in enumerate(shape):
        dim_len = int(dim_len)
        if dim >= len(sharding.dim_specs) or not sharding.dim_specs[dim].axes:
            slices.append(slice(None))
            continue
        
        # Compute position for multi-axis sharding
        dim_spec = sharding.dim_specs[dim]
        total, pos = 1, 0
        for axis in dim_spec.axes:
            size = sharding.mesh.get_axis_size(axis)
            pos = pos * size + sharding.mesh.get_coordinate(shard_idx, axis)
            total *= size
        
        chunk = math.ceil(dim_len / total)
        start = min(pos * chunk, dim_len)
        slices.append(slice(start, min(start + chunk, dim_len)))
    
    return tensor_val[tuple(slices)]


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
    
    For already-sharded inputs: use their shard values.
    For replicated inputs: slice according to that input's propagated sharding.
    
    Args:
        args: Input arguments (may contain Tensors)
        shard_idx: Index of the shard to extract
        per_input_shardings: List of ShardingSpecs, one per input tensor
        g: graph module
        Tensor: Tensor class
        pytree: pytree module
    """
    from typing import List
    
    input_idx = [0]  # Use list for closure mutation
    
    def extract(x):
        if not isinstance(x, Tensor):
            return x
        
        # Get THIS input's sharding (may differ from other inputs)
        this_sharding = None
        if input_idx[0] < len(per_input_shardings):
            this_sharding = per_input_shardings[input_idx[0]]
        input_idx[0] += 1
        
        vals = ensure_shard_values(x)
        candidate = None
        
        if len(vals) > 1 and shard_idx < len(vals):
            # We have specific value for this shard (could be replicated or sharded)
            candidate = vals[shard_idx]
        else:
            # Fallback to global value (or lazy load)
            candidate = g.TensorValue(x)
        
        # If we need sharding, check if we need to slice
        if this_sharding and not this_sharding.is_fully_replicated():
            # If candidate shape matches global shape, we imply it's full data (replicated)
            # and needs to be sliced to match the target sharding.
            # NOTE: We use strict equality check on known dimensions.
            if tuple(candidate.type.shape) == tuple(x.shape):
                 return slice_for_shard(candidate, x.shape, this_sharding, shard_idx)
        
        return candidate
    
    return pytree.tree_map(extract, args)






# ============================================================================
# Resharding
# ============================================================================

def reshard_tensor(tensor: "Tensor", from_spec: Optional["ShardingSpec"],
                   to_spec: Optional["ShardingSpec"], mesh: "DeviceMesh") -> "Tensor":
    """Reshard tensor from one sharding spec to another.
    
    Strategy: Gather all sharded axes to get global data, then shard according to target.
    This is simpler and more robust than incremental gather/shard.
    """
    from ..ops.communication import all_gather, shard as shard_op
    
    if mesh is None:
        return tensor
        
    if not from_spec:
        from_spec = create_replicated_spec(mesh, len(tensor.shape))
        
    if not needs_reshard(from_spec, to_spec):
        return tensor

    # Step 1: Gather all axes to get fully replicated (global) tensor
    # We iterate over the *current* sharding spec (from_spec)
    # and all_gather any dimension that is sharded.
    result = tensor
    if from_spec:
        for dim, dim_spec in enumerate(from_spec.dim_specs):
            # If this dimension is sharded (has axes), gather it
            if dim_spec.axes:
                 # Note: all_gather adds 'Replication' semantics to this dim
                 # but keeps the Tensor rank same.
                 # The resulting tensor has updated sharding (replicated on this dim).
                 result = all_gather(result, axis=dim)
    
    # Step 2: Apply target sharding (shard_op works on global data)
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
