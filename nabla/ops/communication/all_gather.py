# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List

from max.graph import TensorValue, ops

from ..base import Operation
from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
    from ...core import Tensor


class AllGatherOp(CollectiveOperation):
    """Gather shards along an axis to produce replicated full tensors."""
    
    @property
    def name(self) -> str:
        return "all_gather"

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: "DeviceMesh",
        axes: list[str],
        input_specs: list["ShardingSpec"] = None,
        output_specs: list["ShardingSpec"] = None,
    ) -> float:
        """Estimate AllGather cost."""
        # For AllGather, size_bytes usually represents the output (full) size.
        # But we need to know the LOCAL shard size for bandwidth calc.
        # Approximation: if input_specs provided, use explicit shard info.
        
        if not axes:
            return 0.0
            
        n_devices = 1
        for axis in axes:
            n_devices *= mesh.get_axis_size(axis)
        
        if n_devices <= 1:
            return 0.0
            
        # If size_bytes is total size, local is size_bytes / n_devices
        local_bytes = size_bytes // n_devices
        
        bandwidth = getattr(mesh, 'bandwidth', 1.0)
        # Cost = (N-1)/N * TotalSize / Bandwidth
        # Or: (N-1) * LocalSize / Bandwidth
        cost = (n_devices - 1) / n_devices * size_bytes / bandwidth
        return cost
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        axis: int,
        mesh: "DeviceMesh" = None,
        sharded_axis_name: str = None,
    ) -> List[TensorValue]:
        """Gather all shards along the specified axis."""
        # DISTRIBUTED: Use native MAX allgather
        if mesh and mesh.is_distributed:
            from max.graph.ops.allgather import allgather as max_allgather
            from max.graph.type import BufferType
            from max.dtype import DType
            
            # Create signal buffers (one per device)
            signal_buffers = [
                ops.buffer_create(BufferType(DType.int64, (1,), dev))
                for dev in mesh.device_refs
            ]
            return max_allgather(shard_values, signal_buffers, axis=axis)
        
        # SIMULATED
        if mesh is None or sharded_axis_name is None:
            # Fallback: simple concat all
            if len(shard_values) == 1:
                return shard_values
            full_tensor = ops.concat(shard_values, axis=axis)
            return [full_tensor] * len(shard_values)
        
        return self._simulate_grouped_gather(shard_values, axis, mesh, sharded_axis_name)

    def _simulate_grouped_gather(self, shard_values, axis, mesh, sharded_axis_name):
        """Handle simulated gather logic for multi-dimensional meshes."""
        # Get all mesh axis names
        all_axes = list(mesh.axis_names)
        other_axes = [ax for ax in all_axes if ax != sharded_axis_name]
        
        if not other_axes:
            # 1D mesh: simple case - gather all
            # Group by coordinate on the sharded axis to ensure correct ordering
            unique_shards = []
            seen_coords = set()
            for shard_idx, val in enumerate(shard_values):
                coord = mesh.get_coordinate(shard_idx, sharded_axis_name)
                if coord not in seen_coords:
                    seen_coords.add(coord)
                    unique_shards.append((coord, val))
            unique_shards.sort(key=lambda x: x[0])
            shards_to_concat = [v for _, v in unique_shards]
            
            full_tensor = ops.concat(shards_to_concat, axis=axis) if len(shards_to_concat) > 1 else shards_to_concat[0]
            return [full_tensor] * len(shard_values)
        
        # Multi-dimensional: Group devices by their coordinates on OTHER axes
        groups = self._group_shards_by_axes(shard_values, mesh, other_axes)
        
        # Gather per group
        gathered_per_group = {}
        for other_coords, members in groups.items():
            # Sort by coord on the sharded axis
            members.sort(key=lambda x: mesh.get_coordinate(x[0], sharded_axis_name))
            shards = [val for _, val in members]
            gathered = ops.concat(shards, axis=axis) if len(shards) > 1 else shards[0]
            gathered_per_group[other_coords] = gathered
        
        # Distribute results
        results = []
        for shard_idx in range(len(shard_values)):
            other_coords = tuple(mesh.get_coordinate(shard_idx, ax) for ax in other_axes)
            results.append(gathered_per_group[other_coords])
        return results
    
    def __call__(self, sharded_tensor, axis: int, **kwargs):
        """Gather all shards to produce a replicated tensor.
        
        Args:
            sharded_tensor: Tensor with multiple shards
            axis: Axis along which shards are split
            
        Returns:
            Replicated tensor with gathered values
        """
        from ...core import Tensor
        from ...core import TensorImpl
        from ...core import GRAPH
        from ...core.sharding.spec import ShardingSpec, DimSpec
        
        if (not sharded_tensor._impl._values and not sharded_tensor._impl._storages) or \
           (sharded_tensor._impl.sharding and sharded_tensor._impl.sharding.is_fully_replicated()):
            return sharded_tensor  # Already replicated or no data
            
        # Get mesh and sharded axis info from input sharding
        mesh = sharded_tensor._impl.sharding.mesh if sharded_tensor._impl.sharding else None
        sharded_axis_name = None
        
        if sharded_tensor._impl.sharding:
            # Find which mesh axis this tensor dimension is sharded on
            sharding = sharded_tensor._impl.sharding
            if axis < len(sharding.dim_specs) and sharding.dim_specs[axis].axes:
                sharded_axis_name = sharding.dim_specs[axis].axes[0]
        
        # Hydrate values from storages if needed (realized tensor)
        sharded_tensor.hydrate()
            
        if len(sharded_tensor.values) <= 1:
            # Physically gathered (single value) but logically sharded.
            # We just need to update the metadata to be replicated.
            # IMPORTANT: Compute the GLOBAL shape, not just copy local shape!
            from ...core.sharding.spec import compute_global_shape
            from max.graph import Shape
            
            batch_dims = sharded_tensor._impl.batch_dims
            
            # Get local physical shape
            local_shape = sharded_tensor._impl.physical_local_shape(0)
            if local_shape is None:
                local_shape = sharded_tensor.shape  # Fallback
            
            # Compute global shape from local shape and current sharding
            if sharded_tensor._impl.sharding and local_shape is not None:
                global_shape_tuple = compute_global_shape(tuple(local_shape), sharded_tensor._impl.sharding)
                global_shape = Shape(global_shape_tuple)
            else:
                global_shape = local_shape
            
            # Create replicated output sharding spec with correct rank
            rank = len(global_shape) if global_shape else len(sharded_tensor.shape)
            replicated_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)]) if mesh else None
            
            impl = TensorImpl(
                storages=sharded_tensor._impl._storages,  # Copy storages!
                values=sharded_tensor._impl._values,  # Keep raw for passthrough
                traced=sharded_tensor._impl.traced,
                batch_dims=batch_dims,
            )
            impl.sharding = replicated_spec
            # NABLA 2026: Cached metadata removed.
            
            return Tensor(impl=impl)
        
        with GRAPH.graph:
            gathered = self.maxpr(
                sharded_tensor.values, axis, 
                mesh=mesh, sharded_axis_name=sharded_axis_name
            )
        
        # Compute global shape from input: after gather on axis, that axis is replicated
        from ...core.sharding.spec import compute_global_shape
        from max.graph import Shape
        
        local_shape = sharded_tensor._impl.physical_local_shape(0)
        if sharded_tensor._impl.sharding and local_shape is not None:
            global_shape_tuple = compute_global_shape(tuple(local_shape), sharded_tensor._impl.sharding)
            global_shape = Shape(global_shape_tuple)
        else:
             # Fallback if no local shape info
             global_shape = None
        
        # Create output sharding spec: only the gathered dimension becomes replicated
        # Other dimensions keep their original sharding
        output_spec = None
        if mesh and sharded_tensor._impl.sharding:
            input_spec = sharded_tensor._impl.sharding
            new_dim_specs = []
            for dim_idx, dim_spec in enumerate(input_spec.dim_specs):
                if dim_idx == axis:
                    # This dimension is now gathered/replicated
                    new_dim_specs.append(DimSpec([]))
                else:
                    # Preserve original sharding
                    new_dim_specs.append(dim_spec)
            output_spec = ShardingSpec(mesh, new_dim_specs)
        
        # Create output tensor with global shape info computed dynamically
        impl = TensorImpl(
            values=gathered,
            traced=sharded_tensor._impl.traced,
            batch_dims=sharded_tensor._impl.batch_dims,
        )
        impl.sharding = output_spec
        # NABLA 2026: Cached metadata removed.
        output = Tensor(impl=impl)
        
        # Setup tracing refs for graph traversal
        self._setup_output_refs(output, (sharded_tensor,), {'axis': axis}, sharded_tensor._impl.traced)
        
        return output


class GatherAllAxesOp(Operation):
    """Gather all sharded axes to produce a fully replicated tensor.
    
    Uses hierarchical concatenation to properly handle multi-axis sharding.
    This is the correct way to reconstruct a tensor sharded on multiple dimensions.
    """
    
    @property
    def name(self) -> str:
        return "gather_all_axes"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        source_spec: "ShardingSpec",
    ) -> TensorValue:
        """Reconstruct the global tensor from potentially multi-dimensional shards.
        
        Algorithm (hierarchical concatenation):
        1. For each sharded tensor dimension (reverse order):
           a. Group shards by coordinates on ALL mesh axes EXCEPT the one being merged
           b. Within each group, sort by coordinate on the merge axis
           c. Concatenate group members along the tensor dimension
        2. After processing all sharded dimensions, one global tensor remains
        """
        # from max.graph import ops (already imported globally)
        
        # If duplicated/replicated, any shard is the global tensor
        if source_spec.is_fully_replicated():
            return shard_values[0]

        # Build (value, device_id) descriptors for hierarchical concatenation
        current_shard_descs = [
            (shard_values[i], source_spec.mesh.devices[i]) 
            for i in range(len(shard_values))
        ]
        rank = len(shard_values[0].type.shape)
        
        # Track active axes for distinguishing shards
        current_active_axes = set()
        for dim in source_spec.dim_specs:
            current_active_axes.update(dim.axes)
        current_active_axes.update(source_spec.replicated_axes)
        
        # Process dimensions in reverse order
        for d in range(rank - 1, -1, -1):
            if d >= len(source_spec.dim_specs): continue
            dim_spec = source_spec.dim_specs[d]
            
            for ax in reversed(dim_spec.axes):
                # Group shards by coordinates on all axes except the merge axis
                groups = {}
                for val, device_id in current_shard_descs:
                    signature = []
                    # Key depends on all currently active axes EXCEPT the one we are merging
                    for check_ax in sorted(list(current_active_axes)):
                        if check_ax == ax:
                            continue
                        # Use computed coord
                        try:
                            c = source_spec.mesh.get_coordinate(device_id, check_ax)
                            signature.append((check_ax, c))
                        except Exception:
                            # Should not happen if spec is valid
                            continue
                            
                    key = tuple(signature)
                    
                    if key not in groups:
                        groups[key] = []
                    
                    # Store (coord_on_ax, val, device_id)
                    my_coord = source_spec.mesh.get_coordinate(device_id, ax)
                    groups[key].append((my_coord, val, device_id))
                
                new_shard_descs = []
                # Process each group
                for key, members in groups.items():
                    # Sort by coord on `ax`
                    members.sort(key=lambda x: x[0])
                    
                    # Filter unique coords (handle replication)
                    # If multiple devices have same coord on `ax` and same Key, they are replicas.
                    unique_chunks = []
                    seen_coords = set()
                    
                    for m in members:
                        coord = m[0]
                        if coord not in seen_coords:
                            unique_chunks.append(m[1])
                            seen_coords.add(coord)
                    
                    merged = ops.concat(unique_chunks, axis=d)
                    
                    # keep representative device_id from first member
                    new_shard_descs.append((merged, members[0][2]))
                
                current_shard_descs = new_shard_descs
                current_active_axes.remove(ax)

        return current_shard_descs[0][0]

    def __call__(self, sharded_tensor):
        """Gather all sharded axes to produce a replicated tensor.
        
        Args:
            sharded_tensor: Tensor with any sharding configuration
            
        Returns:
            Replicated tensor with the full global data
        """
        from ...core import Tensor
        from ...core import TensorImpl
        from ...core import GRAPH
        from ...core.sharding.spec import ShardingSpec, DimSpec
        from max import graph as g
        
        if not sharded_tensor._impl.sharding:
            return sharded_tensor  # No sharding info
        
        spec = sharded_tensor._impl.sharding
        mesh = spec.mesh
        
        if spec.is_fully_replicated():
            return sharded_tensor  # Already replicated
        
        # Hydrate values from storages if needed (realized tensor)
        sharded_tensor.hydrate()
        shard_values = sharded_tensor.values
        
        if not shard_values or len(shard_values) <= 1:
            return sharded_tensor  # Nothing to gather
        
        with GRAPH.graph:
            global_tensor = self.maxpr(shard_values, spec)
        
        # Create replicated output
        rank = len(global_tensor.type.shape)
        replicated_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])
        
        impl = TensorImpl(
            values=[global_tensor],
            traced=sharded_tensor._impl.traced,
            batch_dims=sharded_tensor._impl.batch_dims,
        )
        impl.sharding = replicated_spec
        return Tensor(impl=impl)


# Singleton instances
all_gather_op = AllGatherOp()
gather_all_axes_op = GatherAllAxesOp()

# Public API functions
def all_gather(sharded_tensor, axis: int, **kwargs):
    """Gather all shards to produce a replicated tensor.
    
    Args:
        sharded_tensor: Tensor with multiple shards
        axis: Axis along which shards are split
        **kwargs: Additional arguments for internal use
        
    Returns:
        Replicated tensor with gathered values
    """
    return all_gather_op(sharded_tensor, axis, **kwargs)


def gather_all_axes(sharded_tensor):
    """Gather all sharded axes to produce a fully replicated tensor.
    
    Args:
        sharded_tensor: Tensor with any sharding configuration
        
    Returns:
        Replicated tensor with the full global data
    """
    return gather_all_axes_op(sharded_tensor)
