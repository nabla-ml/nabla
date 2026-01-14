# ===----------------------------------------------------------------------=== #
# Nabla 2026
# ===----------------------------------------------------------------------=== #

"""Communication Cost Model for Sharding Optimization.

This module provides cost estimation functions for collective operations
(AllReduce, AllGather, ReduceScatter) and resharding operations. These
costs enable objective-based sharding optimization in the SimpleSolver.

The cost model uses normalized bandwidth (default 1.0), making costs
relative rather than absolute. This allows correct DP vs MP tradeoffs
without requiring real hardware profiling.

Cost Formulas (Ring Algorithm):
- AllReduce: 2 * (n-1)/n * size_bytes (reduce-scatter + all-gather)
- AllGather: (n-1)/n * size_bytes
- ReduceScatter: (n-1)/n * size_bytes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from .spec import DeviceMesh, ShardingSpec


def allreduce_cost(
    size_bytes: int,
    mesh: "DeviceMesh",
    axes: List[str],
) -> float:
    """Estimate cost of AllReduce across specified mesh axes.
    
    Uses ring AllReduce formula: 2 * (n-1)/n * size_bytes
    This represents reduce-scatter + all-gather phases.
    
    Args:
        size_bytes: Size of tensor data in bytes
        mesh: DeviceMesh to communicate across
        axes: Mesh axes to reduce over
        
    Returns:
        Estimated communication cost (normalized)
    """
    if not axes:
        return 0.0
    
    # Calculate total devices participating in the reduction
    n_devices = 1
    for axis in axes:
        n_devices *= mesh.get_axis_size(axis)
    
    if n_devices <= 1:
        return 0.0
    
    # Ring AllReduce cost: 2 * (n-1)/n * size
    # Factor of 2 accounts for reduce-scatter + all-gather
    bandwidth = getattr(mesh, 'bandwidth', 1.0)
    cost = 2.0 * (n_devices - 1) / n_devices * size_bytes / bandwidth
    
    return cost


def allgather_cost(
    size_bytes: int,
    mesh: "DeviceMesh",
    axes: List[str],
) -> float:
    """Estimate cost of AllGather across specified mesh axes.
    
    Uses ring AllGather formula: (n-1)/n * total_size
    where total_size = size_bytes * n_devices (gathering from all devices).
    
    Args:
        size_bytes: Size of LOCAL tensor shard in bytes
        mesh: DeviceMesh to communicate across
        axes: Mesh axes to gather over
        
    Returns:
        Estimated communication cost (normalized)
    """
    if not axes:
        return 0.0
    
    n_devices = 1
    for axis in axes:
        n_devices *= mesh.get_axis_size(axis)
    
    if n_devices <= 1:
        return 0.0
    
    # Ring AllGather cost: (n-1)/n * total_gathered_size
    # Each device sends (n-1) * local_size in ring pattern
    bandwidth = getattr(mesh, 'bandwidth', 1.0)
    cost = (n_devices - 1) / n_devices * size_bytes * n_devices / bandwidth
    
    return cost


def reduce_scatter_cost(
    size_bytes: int,
    mesh: "DeviceMesh",
    axes: List[str],
) -> float:
    """Estimate cost of ReduceScatter across specified mesh axes.
    
    Uses ring ReduceScatter formula: (n-1)/n * size_bytes
    (same as AllGather for ring algorithm).
    
    Args:
        size_bytes: Size of tensor data in bytes (before scatter)
        mesh: DeviceMesh to communicate across
        axes: Mesh axes to reduce-scatter over
        
    Returns:
        Estimated communication cost (normalized)
    """
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


def resharding_cost(
    from_spec: Optional["ShardingSpec"],
    to_spec: Optional["ShardingSpec"],
    tensor_bytes: int,
    mesh: "DeviceMesh",
) -> float:
    """Estimate cost of resharding a tensor from one spec to another.
    
    Resharding involves:
    - AllGather for dimensions that become less sharded
    - Shard (local slice) for dimensions that become more sharded
    
    Only AllGather contributes communication cost; local slicing is free.
    
    Args:
        from_spec: Current sharding specification
        to_spec: Target sharding specification
        tensor_bytes: Total tensor size in bytes (global)
        mesh: DeviceMesh for communication
        
    Returns:
        Estimated communication cost (normalized)
    """
    # Default: no resharding if specs are None or identical
    if from_spec is None and to_spec is None:
        return 0.0
    
    if from_spec is None:
        # Unsharded -> Sharded: just local slicing, no communication
        return 0.0
    
    if to_spec is None:
        # Sharded -> Unsharded: need AllGather on all sharded dims
        axes_to_gather: Set[str] = set()
        for dim_spec in from_spec.dim_specs:
            axes_to_gather.update(dim_spec.axes)
        return allgather_cost(
            tensor_bytes // _get_total_shards(from_spec),
            mesh,
            list(axes_to_gather)
        )
    
    # Compare dimension-by-dimension
    total_cost = 0.0
    
    if len(from_spec.dim_specs) != len(to_spec.dim_specs):
        # Rank mismatch - can't reshard directly
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
            local_bytes = tensor_bytes // from_shards
            
            total_cost += allgather_cost(local_bytes, mesh, list(removed_axes))
    
    return total_cost


def _get_total_shards(spec: "ShardingSpec") -> int:
    """Get total number of shards across all dimensions."""
    total = 1
    for dim_spec in spec.dim_specs:
        for axis in dim_spec.axes:
            total *= spec.mesh.get_axis_size(axis)
    return total
