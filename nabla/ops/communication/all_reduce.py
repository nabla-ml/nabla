# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Set

from max.graph import TensorValue, ops

from .base import CollectiveOperation
from .all_gather import AllGatherOp

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec


class AllReduceOp(CollectiveOperation):
    """Reduce values across all shards using the specified reduction."""
    
    @property
    def name(self) -> str:
        return "all_reduce"

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: "DeviceMesh",
        axes: list[str],
        input_specs: list["ShardingSpec"] = None,
        output_specs: list["ShardingSpec"] = None,
    ) -> float:
        """Estimate AllReduce cost."""
        if not axes:
            return 0.0
            
        n_devices = 1
        for axis in axes:
            n_devices *= mesh.get_axis_size(axis)
        
        if n_devices <= 1:
            return 0.0
            
        bandwidth = getattr(mesh, 'bandwidth', 1.0)
        # 2 * (N-1)/N * Size / Bandwidth
        cost = 2.0 * (n_devices - 1) / n_devices * size_bytes / bandwidth
        return cost
    
    def _should_proceed(self, tensor):
        """Check if all_reduce should proceed."""
        # Check both values and storages for multi-shard tensors
        has_multiple_shards = (
            (tensor._impl._values and len(tensor._impl._values) > 1) or
            (tensor._impl._storages and len(tensor._impl._storages) > 1)
        )
        if not has_multiple_shards:
            return False
        
        # IDEMPOTENCY: If tensor is already fully replicated, skip reduction
        if tensor._impl.sharding and tensor._impl.sharding.is_fully_replicated():
            return False
        
        return True
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        mesh: "DeviceMesh" = None,
    ) -> List[TensorValue]:
        """Sum-reduce across all shards (AllReduce)."""
        if not shard_values:
            return []
        
        # DISTRIBUTED: Use native MAX allreduce
        if mesh and mesh.is_distributed:
            from max.graph.ops.allreduce import sum as allreduce_sum
            from max.graph.type import BufferType
            from max.dtype import DType
            
            # Create signal buffers (one per device)
            signal_buffers = [
                ops.buffer_create(BufferType(DType.int64, (1,), dev))
                for dev in mesh.device_refs
            ]
            return allreduce_sum(shard_values, signal_buffers)
        
        # SIMULATED: Sum using loop-based fallback
        result = shard_values[0]
        for sv in shard_values[1:]:
            result = ops.add(result, sv)
        
        # All shards get the same reduced value
        return [result] * len(shard_values)
    
    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output is fully replicated."""
        from ...core.sharding.spec import ShardingSpec, DimSpec
        mesh = input_tensor._impl.sharding.mesh if input_tensor._impl.sharding else None
        
        if mesh and results:
            rank = len(results[0].type.shape)
            return ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])
        return None

    def simulate_grouped_execution(
        self,
        shard_results: List[TensorValue], 
        mesh: "DeviceMesh", 
        reduce_axes: "Set[str]",
    ) -> List[TensorValue]:
        """Simulate grouped AllReduce execution for SPMD verification."""
        if not reduce_axes:
            return shard_results
            
        num_shards = len(shard_results)
        
        # Check if we're reducing over all axes (simple case)
        all_axes = set(mesh.axis_names)
        if all_axes.issubset(reduce_axes):
            return self.maxpr(shard_results, mesh=mesh)
            
        # Group by axes NOT in reduce_axes (we reduce WITHIN these groups)
        group_axes = [ax for ax in all_axes if ax not in reduce_axes]
        groups = self._group_shards_by_axes(shard_results, mesh, group_axes)
        
        # Execute reduction per group
        new_results = [None] * num_shards
        
        for key, group_members in groups.items():
            # group_members is a list of (shard_idx, shard_value)
            group_shards = [val for _, val in group_members]
            
            if len(group_shards) > 1:
                curr_reduced = self.maxpr(group_shards, mesh=mesh)
            else:
                curr_reduced = [group_shards[0]]
                
            # Distribute results back to original positions
            for i, (shard_idx, _) in enumerate(group_members):
                new_results[shard_idx] = curr_reduced[i] if isinstance(curr_reduced, list) else curr_reduced
                
        return new_results


class PMeanOp(CollectiveOperation):
    """Compute mean across all shards (psum / axis_size).
    
    This is a convenience wrapper that combines psum with division
    by the number of devices.
    """
    
    @property
    def name(self) -> str:
        return "pmean"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        mesh: "DeviceMesh" = None,
        axis_name: str = None,
    ) -> List[TensorValue]:
        """Compute mean across shards.
        
        Args:
            shard_values: List of shard TensorValues
            mesh: Device mesh
            axis_name: Name of axis to reduce along (for size calculation)
            
        Returns:
            List of mean TensorValues (all identical)
        """
        # First do psum
        reduced = all_reduce_op.maxpr(shard_values, mesh=mesh)
        
        # Then divide by axis size
        if axis_name and mesh:
            axis_size = mesh.get_axis_size(axis_name)
        else:
            axis_size = len(shard_values)
        
        # Divide each result by axis_size
        # Get dtype and device from the reduced tensor
        dtype = reduced[0].type.dtype
        device = reduced[0].type.device
        scale = ops.constant(1.0 / axis_size, dtype, device)
        return [ops.mul(r, scale) for r in reduced]
    
    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output is fully replicated."""
        from ...core.sharding.spec import ShardingSpec, DimSpec
        mesh = input_tensor._impl.sharding.mesh if input_tensor._impl.sharding else None
        
        if mesh and results:
            rank = len(results[0].type.shape)
            return ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])
        return None


# Singleton instances
all_reduce_op = AllReduceOp()
pmean_op = PMeanOp()

# Public API functions
def all_reduce(sharded_tensor, **kwargs):
    """Sum-reduce across all shards.
    
    Note: MAX only supports sum reduction natively.
    
    Args:
        sharded_tensor: Tensor with partial values per shard
        **kwargs: Additional arguments
        
    Returns:
        Tensor with sum-reduced values (replicated across shards)
    """
    return all_reduce_op(sharded_tensor, **kwargs)


def pmean(sharded_tensor, axis_name: str = None):
    """Compute mean across all shards.
    
    Equivalent to psum(x) / axis_size. Useful for averaging gradients
    in data parallelism.
    
    Args:
        sharded_tensor: Tensor with partial values per shard
        axis_name: Name of mesh axis for size calculation (optional)
        
    Returns:
        Tensor with mean values (replicated across shards)
    """
    return pmean_op(sharded_tensor, axis_name=axis_name)
