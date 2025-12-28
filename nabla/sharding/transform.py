"""Transform utilities for sharding (re-exports for backward compatibility).

The transform_trace_for_sharding function is DEPRECATED - eager sharding now
happens automatically in Operation.__call__ when sharded inputs are detected.

This module provides backward-compatible imports for existing tests.
"""

from ..core.tensor_impl import get_topological_order
from .propagation import propagate_sharding, OpShardingRule


def run_propagation(op_impls: list, mesh):
    """Run sharding propagation over a list of operation impls.
    
    DEPRECATED: Eager sharding now happens automatically in Operation.__call__.
    This function is provided for backward compatibility with tests.
    """
    from .spec import ShardingSpec, DimSpec
    
    for impl in op_impls:
        if impl.op is None:
            continue
        
        # Skip if already has sharding
        if impl.sharding is not None:
            continue
        
        # Get parent shardings
        parent_specs = []
        input_shapes = []
        for parent in impl.parents:
            if parent.sharding is not None:
                parent_specs.append(parent.sharding)
                if parent.cached_shape:
                    input_shapes.append(tuple(int(d) for d in parent.cached_shape))
                elif parent.physical_shape:
                    input_shapes.append(tuple(int(d) for d in parent.physical_shape))
        
        if not parent_specs:
            continue
        
        # Get sharding rule from operation
        rule = impl.op.sharding_rule(input_shapes, input_shapes[:1]) if input_shapes else None
        
        if rule is None:
            # Copy sharding from first sharded parent
            impl.sharding = parent_specs[0]
            continue
        
        # Create output spec and propagate
        output_rank = len(input_shapes[0]) if input_shapes else 0
        output_spec = ShardingSpec(mesh, [DimSpec([], is_open=True) for _ in range(output_rank)])
        
        propagate_sharding(rule, parent_specs, [output_spec])
        impl.sharding = output_spec


def transform_trace_for_sharding(root_impls: list, mesh):
    """Transform a traced computation for sharded execution.
    
    DEPRECATED: This function is no longer needed. Eager sharding now happens
    automatically in Operation.__call__ when sharded inputs are detected.
    Each operation's _call_spmd method handles per-shard execution.
    
    This function is provided for backward compatibility but does nothing
    useful in the new eager sharding architecture.
    """
    # In the new architecture, sharding happens eagerly.
    # This function is a no-op placeholder for backward compatibility.
    pass


__all__ = [
    "get_topological_order",
    "run_propagation",
    "transform_trace_for_sharding",
]
