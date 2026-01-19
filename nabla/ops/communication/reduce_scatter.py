# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List

from max.graph import TensorValue, ops

from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec


class ReduceScatterOp(CollectiveOperation):
    """Reduce then scatter the result across shards."""
    
    @property
    def name(self) -> str:
        return "reduce_scatter"

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: "DeviceMesh",
        axes: list[str],
        input_specs: list["ShardingSpec"] = None,
        output_specs: list["ShardingSpec"] = None,
    ) -> float:
        if not axes:
            return 0.0
            
        n_devices = 1
        for axis in axes:
            n_devices *= mesh.get_axis_size(axis)
        
        if n_devices <= 1:
            return 0.0
            
        bandwidth = getattr(mesh, 'bandwidth', 1.0)
        cost = (n_devices - 1) / n_devices * size_bytes / bandwidth
        return cost
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        axis: int,
        mesh: "DeviceMesh" = None,
    ) -> List[TensorValue]:
        """Sum-reduce across shards then scatter the result."""
        # DISTRIBUTED: Compose allreduce + scatter
        if mesh and mesh.is_distributed:
            from max.graph.ops.allreduce import sum as allreduce_sum
            from max.graph.type import BufferType
            from max.dtype import DType
            
            # Create signal buffers
            signal_buffers = [
                ops.buffer_create(BufferType(DType.int64, (1,), dev))
                for dev in mesh.device_refs
            ]
            
            # Reduce across all shards
            reduced_values = allreduce_sum(shard_values, signal_buffers)
            
            # Scatter: each shard gets a slice of the result
            num_shards = len(shard_values)
            shape = reduced_values[0].type.shape
            axis_size = int(shape[axis])
            chunk_size = axis_size // num_shards
            
            scattered = []
            for i, rv in enumerate(reduced_values):
                slices = [slice(None)] * len(shape)
                slices[axis] = slice(i * chunk_size, (i + 1) * chunk_size)
                scattered.append(rv[tuple(slices)])
            
            return scattered
        
        # SIMULATED: Sum all values then scatter
        full_result = shard_values[0]
        for sv in shard_values[1:]:
            full_result = ops.add(full_result, sv)
        
        # Then scatter (split) along the axis
        num_shards = len(shard_values)
        shape = full_result.type.shape
        axis_size = int(shape[axis])
        chunk_size = axis_size // num_shards
        
        scattered = []
        for i in range(num_shards):
            # Create slice for this shard
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(i * chunk_size, (i + 1) * chunk_size)
            scattered.append(full_result[tuple(slices)])
        
        return scattered
    
    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output sharding: the scatter axis becomes sharded."""
        from ...core.sharding.spec import ShardingSpec, DimSpec
        
        mesh = input_tensor.sharding.mesh if input_tensor.sharding else None
        input_spec = input_tensor.sharding
        
        if mesh and input_spec:
            # Create new spec where the scatter axis is sharded
            # Note: This is a simplification. Real logic would depend on which mesh axis we scattered over.
            # But currently ReduceScatterOp takes an int axis, implying we scatter over the *shards*.
            # If the shards correspond to a mesh axis, we should mark it.
            # For now, we logic similar to original implementation which assumed appending new dim specs.
            
            # Use scattered results to check rank
            rank = len(results[0].type.shape) if results else 0
            new_dim_specs = []
            for d in range(rank):
                if d < len(input_spec.dim_specs):
                    new_dim_specs.append(input_spec.dim_specs[d])
                else:
                    new_dim_specs.append(DimSpec([]))
            return ShardingSpec(mesh, new_dim_specs)
            
        return None

# Singleton instances
reduce_scatter_op = ReduceScatterOp()

# Public API functions
def reduce_scatter(sharded_tensor, axis: int, **kwargs):
    """Sum-reduce then scatter result across shards.
    
    Note: MAX only supports sum reduction natively.
    """
    return reduce_scatter_op(sharded_tensor, axis=axis, **kwargs)
