# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional, Set

from max.graph import TensorValue, ops

from ..base import Operation
from .all_gather import AllGatherOp, all_gather
from .shard import shard_op

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
    from ...core import Tensor


class ReshardOp(Operation):
    """Generic resharding operation.
    
    Reshards a tensor from its current sharding (or replication) to a new target
    sharding specification. Handles both logical to physical spec conversion
    (for batch_dims) and the actual data movement (gather + shard).
    """
    
    @property
    def name(self) -> str:
        return "reshard"

    def communication_cost(
        self, 
        input_specs: list["ShardingSpec"], 
        output_specs: list["ShardingSpec"], 
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        mesh: "DeviceMesh"
    ) -> float:
        """Estimate cost of resharding."""
        # ReshardOp maps input_specs[0] -> output_specs[0] or explicit target
        
        from_spec = input_specs[0] if input_specs else None
        
        # Ideally output_specs[0] is the target.
        to_spec = output_specs[0] if output_specs else None
        
        if not input_shapes:
            return 0.0
            
        # Total tensor bytes
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        tensor_bytes = num_elements * 4
        
        # Default: no resharding if specs are None or identical
        if from_spec is None and to_spec is None:
            return 0.0
        
        if from_spec is None:
            # Unsharded -> Sharded: just local slicing, no communication
            return 0.0
        
        if to_spec is None:
            # Sharded -> Unsharded: need AllGather on all sharded dims
            axes_to_gather = set()
            for dim_spec in from_spec.dim_specs:
                axes_to_gather.update(dim_spec.axes)
            
            # Use AllGatherOp.estimate_cost logic directly
            # Local bytes = Total / shards
            total_shards = from_spec.total_shards
            local_bytes = tensor_bytes // (total_shards or 1)
            
            # Recalculate gathering cost here or delegate? 
            # Delegating is better but AllGatherOp.estimate_cost is static
            return AllGatherOp.estimate_cost(local_bytes, mesh, list(axes_to_gather))
        
        # Compare dimension-by-dimension
        total_cost = 0.0
        
        if len(from_spec.dim_specs) != len(to_spec.dim_specs):
            # Rank mismatch implies reshape or error - infinite cost
            return float('inf')
        
        for from_dim, to_dim in zip(from_spec.dim_specs, to_spec.dim_specs):
            from_axes = set(from_dim.axes)
            to_axes = set(to_dim.axes)
            
            # Axes being removed need AllGather
            removed_axes = from_axes - to_axes
            if removed_axes:
                # Estimate local shard size for this dimension
                from_shards = 1
                for axis in from_dim.axes:
                    from_shards *= mesh.get_axis_size(axis)
                
                # Approximate local size involved in this dim's gather
                local_bytes_dim = tensor_bytes // from_shards
                
                total_cost += AllGatherOp.estimate_cost(local_bytes_dim, mesh, list(removed_axes))
        
        return total_cost
    
    def maxpr(self, *args, **kwargs):
        """ReshardOp is a composite operation that orchestrates other ops.
        
        It doesn't have its own maxpr because resharding is implemented as:
        1. all_gather (if currently sharded) -> get global tensor
        2. shard_op -> apply new sharding
        
        This pattern is similar to JAX's reshard which also compiles to
        a sequence of collectives rather than a single primitive.
        """
        raise NotImplementedError(
            "ReshardOp is a composite operation. "
            "Use __call__ which orchestrates all_gather + shard_op."
        )
        
    def __call__(
        self,
        tensor: "Tensor",
        mesh: "DeviceMesh",
        dim_specs: List["DimSpec"],
        replicated_axes: Optional[Set[str]] = None,
    ) -> "Tensor":
        """Reshard tensor to target specs.
        
        Args:
            tensor: Input tensor
            mesh: Target device mesh
            dim_specs: List of DimSpecs. Can be logical (len=rank) or physical (len=rank+batch_dims).
            replicated_axes: Optional set of axes to force replication on.
        """
        from ...core.sharding.spec import ShardingSpec, DimSpec, needs_reshard
        from ...core.tensor import Tensor
        
        # 1. Handle batch_dims (Logical -> Physical conversion)
        # If tensor has batch_dims, we might need to prepend replicated specs
        batch_dims = tensor._impl.batch_dims
        current_rank = len(tensor.shape) # Logical rank
        
        if batch_dims > 0:
            # Check provided specs length
            if len(dim_specs) == current_rank:
                # User provided logical specs. Prepend replicated batch specs.
                batch_specs = [DimSpec([], is_open=True) for _ in range(batch_dims)]
                
                # Inherit existing batch specs if possible
                if tensor._impl.sharding:
                    current_s = tensor._impl.sharding
                    if len(current_s.dim_specs) >= batch_dims:
                         for i in range(batch_dims):
                             batch_specs[i] = current_s.dim_specs[i].clone()
                             
                dim_specs = batch_specs + list(dim_specs)
            elif len(dim_specs) != (current_rank + batch_dims):
                 # Length mismatch - neither logical nor physical?
                 # Let validation downstream handle it or warn?
                 pass
        
        # 2. Construct Target Spec
        target_spec = ShardingSpec(mesh, dim_specs, replicated_axes=replicated_axes or set())
        current_spec = tensor._impl.sharding
        
        # 3. Check if reshard needed
        if not needs_reshard(current_spec, target_spec):
            if current_spec is None:
                tensor._impl.sharding = target_spec
            return tensor

        # 4. Perform Resharding with SMART per-dimension logic
        # Only gather dimensions where axes are being REMOVED (not extended)
        result = tensor
        if current_spec:
            for dim in range(len(current_spec.dim_specs)):
                from_axes = set(current_spec.dim_specs[dim].axes) if dim < len(current_spec.dim_specs) else set()
                to_axes = set(target_spec.dim_specs[dim].axes) if dim < len(target_spec.dim_specs) else set()
                
                # Only gather if removing axes that aren't preserved in target
                # If from_axes is subset of to_axes, no gather needed (just extending)
                axes_to_remove = from_axes - to_axes
                if axes_to_remove:
                    result = all_gather(result, axis=dim)
        
        # Shard to target using module-level shard_op (efficient)
        result = shard_op(result, mesh, target_spec.dim_specs, replicated_axes=target_spec.replicated_axes)
        
        return result


# Singleton instance
reshard_op = ReshardOp()

# Public API
def reshard(tensor: "Tensor", mesh: "DeviceMesh", dim_specs: List["DimSpec"], replicated_axes: Optional[Set[str]] = None, **kwargs) -> "Tensor":
    """Reshard tensor to target specs.
    
    Args:
        tensor: Input tensor
        mesh: Target device mesh
        dim_specs: Target dimension specs
        replicated_axes: Optional set of axes to replicate over
        **kwargs: Internal arguments
    """
    return reshard_op(tensor, mesh, dim_specs, replicated_axes, **kwargs)
