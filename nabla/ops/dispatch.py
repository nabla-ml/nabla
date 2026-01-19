# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Tuple, List, Optional, Dict

if TYPE_CHECKING:
    from .base import Operation
    from ..core.sharding.spec import ShardingSpec, DeviceMesh



def execute_operation(op: "Operation", *args: Any, **kwargs: Any) -> Any:
    """Unified dispatch for all operations.
    
    Handles sharded and unsharded tensors uniformly via SPMD logic.
    """
    from ..core import Tensor
    from ..core import GRAPH, Tensor

    from ..core import pytree
    from ..core.sharding import spmd
    from max import graph as g
    
    # 1. Collect metadata from inputs
    any_traced = False
    any_has_tangent = False
    max_batch_dims = 0
    any_sharded = False
    
    def collect_metadata(x: Any) -> Any:
        nonlocal any_traced, any_has_tangent, max_batch_dims, any_sharded
        if isinstance(x, Tensor):
            any_traced = any_traced or x.traced
            any_has_tangent = any_has_tangent or (x.tangent is not None)
            max_batch_dims = max(max_batch_dims, x.batch_dims)
            any_sharded = any_sharded or x.is_sharded
        return x
    
    pytree.tree_map(collect_metadata, args)
    
    # 2. Determine execution mode (Implicit)
    mesh = spmd.get_mesh_from_args(args) if any_sharded else None
    
    # 3. Setup for execution (Common)
    # Ensure proper specifications (Pass-through if mesh is None)
    args = spmd.ensure_specs(args, mesh)
    
    # Infer sharding (Returns None/[]/False if mesh is None)
    # Infer sharding (Returns None/[]/False if mesh is None)
    output_sharding, input_shardings, reduce_axes = spmd.infer_output_sharding(
        op, args, mesh, kwargs or {}
    )

    # Eagerly reshard inputs (Pass-through if mesh is None)
    args = spmd.reshard_inputs(args, input_shardings, mesh)
    
    num_shards = len(mesh.devices) if mesh else 1
    
    # 4. Execute operation (Common Loop)
    with GRAPH.graph:
        shard_results = []
        for shard_idx in range(num_shards):
            # Retrieve shard arguments (Handles slicing or value extraction)
            shard_args = spmd.get_shard_args(
                args, shard_idx, input_shardings or [], g, Tensor, pytree
            )
            
            shard_kwargs = op._transform_shard_kwargs(kwargs, output_sharding, shard_idx)
            if shard_kwargs is None:
                shard_kwargs = {}
            shard_results.append(op.maxpr(*shard_args, **shard_kwargs))
    
    # 5. Create output tensors (Unified & Pytree-aware)
    if not shard_results:
        return None # Should not happen
        
    # Reconstruct output structure from first result
    flat_results_per_shard = [pytree.tree_leaves(res) for res in shard_results]
    treedef = pytree.tree_structure(shard_results[0])
    num_leaves = len(flat_results_per_shard[0])
    
    output_leaves = []
    for i in range(num_leaves):
        # Collect this leaf's values across all shards
        leaf_shards = [shard[i] for shard in flat_results_per_shard]
        
        # Create sharded/unsharded tensor from list of values
        tensor = spmd.create_sharded_output(
            leaf_shards, output_sharding, any_traced, max_batch_dims, mesh=mesh
        )
        output_leaves.append(tensor)
        
    output = pytree.tree_unflatten(treedef, output_leaves)
    
    # 7. Common post-processing - set up refs for the main op output
    # This must happen before all_reduce modifies the output
    op._setup_output_refs(output, args, kwargs, any_traced)
    
    
    # 8. AllReduce partial results if contracting dimension was sharded
    if reduce_axes and mesh:
        output = _apply_auto_reduction(op, output, mesh, reduce_axes)
    
    if any_has_tangent:
        _apply_jvp(op, args, output)
    
    return output


def _apply_auto_reduction(op: "Operation", output: Any, mesh: "DeviceMesh", reduce_axes: set[str]) -> Any:
    """Apply automatic AllReduce to partial results if needed."""
    from ..core import Tensor
    from ..core import pytree
    from .communication import all_reduce_op
    from ..core.sharding.spec import ShardingSpec, DimSpec
    from ..core import GRAPH, Tensor


    def apply_grouped_all_reduce(t):
        if not isinstance(t, Tensor):
            return t
        
        # Hydrate values if needed and check we have values to reduce
        t.hydrate()
        if not t._values:  # Check raw after hydrate
            return t
        
        # Apply graph-level grouped all-reduce
        with GRAPH.graph:
            reduced_values = all_reduce_op.simulate_grouped_execution(
                t.values, mesh, reduce_axes
            )
        
        current_spec = t.sharding
        if current_spec:
            new_dim_specs = []
            for ds in current_spec.dim_specs:
                # Remove any axes that were reduce
                new_axes = sorted(list(set(ds.axes) - reduce_axes))
                new_dim_specs.append(DimSpec(new_axes))
            new_spec = ShardingSpec(mesh, new_dim_specs)
        else:
            # Fallback to full replication if no spec (shouldn't happen)
            rank = len(t.shape)
            new_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])

        # Create new Tensor with reduced values
        # Use shared helper to ensure correct global shape metadata and initialization
        from ..core.sharding import spmd
        reduced_tensor = spmd.create_sharded_output(
            reduced_values, new_spec, t.traced, t.batch_dims, mesh
        )
        
        # Setup tracing refs so ALL_REDUCE appears in trace
        trace_kwargs = {'mesh': mesh, 'reduce_axes': list(reduce_axes)}
        all_reduce_op._setup_output_refs(reduced_tensor, (t,), trace_kwargs, t.traced)
        
        return reduced_tensor
        

    return pytree.tree_map(apply_grouped_all_reduce, output)

def _apply_jvp(op: "Operation", args: tuple, output: Any) -> None:
    """Apply JVP rule to compute output tangents."""
    from ..core import Tensor
    from ..core import pytree
    
    tangents = pytree.tree_map(
        lambda x: Tensor(impl=x.tangent) if isinstance(x, Tensor) and x.tangent else None,
        args
    )
    output_tangent = op.jvp_rule(args, tangents, output)
    if output_tangent is not None:
        pytree.tree_map(
            lambda o, t: setattr(o._impl, 'tangent', t._impl) if isinstance(o, Tensor) and isinstance(t, Tensor) else None,
            output, output_tangent
        )
