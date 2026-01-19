# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from ..tensor import Tensor
    from .spec import ShardingSpec
    from .spec import DeviceMesh

# ============================================================================
# Detection & Extraction
# ============================================================================


def get_mesh_from_args(args: tuple) -> Optional["DeviceMesh"]:
    """Extract DeviceMesh from first tensor with sharding spec."""
    from ..tensor import Tensor
    from .. import pytree
    for a in pytree.tree_leaves(args):
        if isinstance(a, Tensor) and a.sharding:
            return a.sharding.mesh
    return None


def ensure_specs(args: tuple, mesh: Optional["DeviceMesh"]) -> tuple:
    """Ensure all tensors have explicit sharding specs (replicated if default)."""
    return args


def reshard_inputs(
    args: tuple, 
    required_specs: "List[Optional[ShardingSpec]]", 
    mesh: Optional["DeviceMesh"]
) -> tuple:
    """Pre-operation resharding: align inputs to required propagation specs."""
    if mesh is None or not required_specs:
        return args
    
    from ..tensor import Tensor
    from .. import pytree
    
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
        
        current = x.sharding
        
        if current is None or current.is_fully_replicated():
            if required is not None and not required.is_fully_replicated():
                from ...ops.communication import shard as shard_fn
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
    """Infer output/input shardings via factor propagation.
    
    Returns:
        (output_sharding, input_shardings, needs_allreduce)
    """
    if mesh is None:
        return None, [], False
        
    # Hook for Ops that define explicit sharding logic (bypassing propagation)
    if hasattr(op, "infer_sharding_spec"):
        return op.infer_sharding_spec(args, mesh, kwargs)

    from ..tensor import Tensor
    from .. import pytree
    from .spec import ShardingSpec, DimSpec
    from .propagation import propagate_sharding
    
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
        spec = t.sharding
        
        shape = t.physical_global_shape
        if shape is None and (t.sharding is None or t.sharding.is_fully_replicated()):
            shape = t.physical_shape
            
        if shape is None:
            batch_dims = t.batch_dims
            if batch_dims > 0 and t.batch_shape is not None:
                batch_ints = tuple(dim_to_int(d) for d in t.batch_shape)
                logical_ints = tuple(dim_to_int(d) for d in t.shape)
                phys_shape_tuple = batch_ints + logical_ints
            else:
                phys_shape_tuple = tuple(dim_to_int(d) for d in t.shape)
        else:
            phys_shape_tuple = tuple(dim_to_int(d) for d in shape)
        
        if spec is None:
            rank = len(phys_shape_tuple)
            spec = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
        input_specs.append(spec.clone())
        input_shapes.append(phys_shape_tuple)
    
    if not any(spec.dim_specs and any(d.axes for d in spec.dim_specs) for spec in input_specs):
        # Even if not sharded on dims, we might have partial sum axes to propagate
        input_partial_axes = set()
        for spec in input_specs:
            input_partial_axes.update(spec.partial_sum_axes)
        
        if not input_partial_axes:
            return None, input_specs, False
        
        output_rank = op.infer_output_rank(input_shapes, **(kwargs or {}))
        output_spec = ShardingSpec(mesh, [DimSpec([], is_open=False) for _ in range(output_rank)], 
                                  partial_sum_axes=input_partial_axes)
        return output_spec, input_specs, set()
    
    # Use rank inference instead of full shape inference
    try:
        output_rank = op.infer_output_rank(input_shapes, **(kwargs or {}))
    except (NotImplementedError, AttributeError):
        output_rank = len(input_shapes[0]) if input_shapes else 0

    # Get sharding rule from operation
    rule = None
    try:
        rule = op.sharding_rule(input_shapes, None, **(kwargs or {}))
    except (NotImplementedError, AttributeError):
        rule = None
        
    if rule is None:
        for spec in input_specs:
            if any(d.axes for d in spec.dim_specs):
                if len(spec.dim_specs) == output_rank:
                    cloned_spec = spec.clone()
                    for dim_spec in cloned_spec.dim_specs:
                        dim_spec.is_open = False
                    return cloned_spec, input_specs, False
        return None, input_specs, False
    
    # Create empty output spec based on rank - MUST BE OPEN to receive propagation
    output_spec = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(output_rank)])
    
    # Run factor-based propagation - updates input_specs and output_spec in-place
    propagate_sharding(rule, input_specs, [output_spec])
    
    # Propagate partial sum axes from inputs to output
    input_partial_axes = set()
    for spec in input_specs:
        if spec:
            input_partial_axes.update(spec.partial_sum_axes)
    output_spec.partial_sum_axes.update(input_partial_axes)
    
    # Check if any contracting factors are sharded (need AllReduce for partial results)
    reduce_axes, ghost_axes = _check_contracting_factors_sharded(rule, input_specs, output_spec)
    
    # Add ghost axes (contracted sharding that didn't trigger AllReduce) to output spec
    for ax in ghost_axes:
        sharded_in_dim = False
        for dim in output_spec.dim_specs:
            if ax in dim.axes:
                dim.partial = True
                sharded_in_dim = True
        
        if not sharded_in_dim:
            output_spec.partial_sum_axes.add(ax)

    # FINAL CLEANUP: Ensure partial sum axes don't overlap with dimension axes
    used_in_dims = set()
    for dim in output_spec.dim_specs:
        for ax in dim.axes:
            used_in_dims.add(ax)
            
    for ax in list(output_spec.partial_sum_axes):
        if ax in used_in_dims:
            output_spec.partial_sum_axes.remove(ax)
            for dim in output_spec.dim_specs:
                if ax in dim.axes:
                    dim.partial = True
    
    # Ensure all output dims are closed for concrete spec
    for dim in output_spec.dim_specs:
        dim.is_open = False
        
    return output_spec, input_specs, reduce_axes


def _check_contracting_factors_sharded(
    rule: "OpShardingRule",
    input_specs: "List[ShardingSpec]",
    output_spec: Optional["ShardingSpec"],
) -> "Tuple[Set[str], Set[str]]":
    """Check if contracting factors need AllReduce (sharded input, not in output)."""
    contracting_factors = rule.get_contracting_factors()
    if not contracting_factors:
        return set(), set()
        
    reduce_axes = set()
    ghost_axes = set()
    
    # Identify which axes are preserved in the output
    preserved_axes = set()
    if output_spec:
        for ds in output_spec.dim_specs:
            preserved_axes.update(ds.axes)

    axis_partial_map: Dict[str, bool] = {}
    contracted_axes = set()
    
    for input_idx, spec in enumerate(input_specs):
        if not spec or input_idx >= len(rule.input_mappings):
            continue
        
        mapping = rule.input_mappings[input_idx]
        
        for dim_idx, current_dim in enumerate(spec.dim_specs):
            factors = mapping.get(dim_idx, [])
            for f in factors:
                if f in contracting_factors:
                    for ax in current_dim.axes:
                        if ax not in preserved_axes:
                            contracted_axes.add(ax)
                            axis_partial_map[ax] = axis_partial_map.get(ax, False) or current_dim.partial
    
    for ax in contracted_axes:
        if axis_partial_map[ax]:
            ghost_axes.add(ax)
        else:
            reduce_axes.add(ax)
                          
    return reduce_axes, ghost_axes


# ============================================================================
# Shape & Slice Computation
# ============================================================================

def create_replicated_spec(mesh: "DeviceMesh", rank: int) -> "ShardingSpec":
    """Create a fully replicated sharding spec."""
    from .spec import ShardingSpec, DimSpec
    return ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])


from .spec import needs_reshard


# ============================================================================
# SPMD Argument Extraction
# ============================================================================


def get_shard_args(args: tuple, shard_idx: int, 
                   per_input_shardings: "List[Optional[ShardingSpec]]",
                   g: Any, Tensor: type, pytree: Any) -> tuple:
    """Get per-shard TensorValues, slicing each input according to its sharding."""
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
        
        x.hydrate()
        vals = x.values
        
        if shard_idx < len(vals):
            return vals[shard_idx]
        return vals[0] if vals else x.__tensorvalue__()
    
    return pytree.tree_map(extract, args)



# ============================================================================
# Resharding
# ============================================================================

def reshard_tensor(tensor: "Tensor", from_spec: Optional["ShardingSpec"],
                   to_spec: Optional["ShardingSpec"], mesh: "DeviceMesh") -> "Tensor":
    """Reshard tensor with minimal communication (smart slicing/gathering)."""
    from ...ops.communication import all_gather, shard as shard_op
    from .spec import DimSpec
    
    if mesh is None:
        return tensor
        
    if not from_spec:
        from_spec = create_replicated_spec(mesh, len(tensor.shape))
        
    if not needs_reshard(from_spec, to_spec):
        return tensor

    result = tensor
    
    # Per-dimension analysis: only gather where axes are being REMOVED
    for dim in range(len(from_spec.dim_specs)):
        from_axes = set(from_spec.dim_specs[dim].axes) if dim < len(from_spec.dim_specs) else set()
        to_axes = set(to_spec.dim_specs[dim].axes) if dim < len(to_spec.dim_specs) else set()
        
        axes_to_remove = from_axes - to_axes
        
        if from_spec.dim_specs[dim].partial:
            from ...ops.communication import all_reduce
            target_is_partial = dim < len(to_spec.dim_specs) and to_spec.dim_specs[dim].partial
            
            if not target_is_partial:
                result = all_reduce(result)
                continue

        if axes_to_remove:
            from ...ops.communication import all_gather
            result = all_gather(result, axis=dim)
    
    for ghost_ax in from_spec.partial_sum_axes:
        if ghost_ax not in to_spec.partial_sum_axes:
            from ...ops.communication import all_reduce
            result = all_reduce(result)

    from ...ops.communication import shard as shard_op
    result = shard_op(result, mesh, to_spec.dim_specs, replicated_axes=to_spec.replicated_axes, _bypass_idempotency=True)
    
    return result





# ============================================================================
# Output Creation
# ============================================================================

def create_sharded_output(results: List[Any], sharding: Optional["ShardingSpec"],
                          traced: bool, batch_dims: int,
                          mesh: Optional["DeviceMesh"] = None) -> "Tensor":
    """Build sharded Tensor from per-shard TensorValues."""
    from ..tensor import Tensor
    from max import graph as g
    
    if not results:
        raise ValueError("Empty shard results")
    
    first = results[0]
    if not isinstance(first, (g.TensorValue, g.BufferValue)):
        return first
    
    if sharding is None and len(results) > 1 and mesh is not None:
        from .spec import ShardingSpec, DimSpec
        rank = len(first.type.shape)
        sharding = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(rank)])
    
    output = Tensor._create_unsafe(
        values=results,
        traced=traced,
        batch_dims=batch_dims,
    )
    output.sharding = sharding
    
    return output
