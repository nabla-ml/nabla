# ===----------------------------------------------------------------------=== #
# Nabla 2026 - Communication Operations for Sharding
# ===----------------------------------------------------------------------=== #

"""Communication operations for distributed/sharded tensor execution.

These operations handle data movement between shards:
- ShardOp: Split a replicated tensor into shards
- AllGatherOp: Gather all shards to produce replicated tensors  
- AllReduceOp: Reduce across shards (sum, mean, etc.)
- ReduceScatterOp: Reduce then scatter result across shards
- ReshardOp: Generic resharding between different sharding specs
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List

from max.graph import TensorValue, ops

from .operation import Operation

if TYPE_CHECKING:
    from ..sharding.spec import DeviceMesh, DimSpec, ShardingSpec


def _is_distributed(mesh: "DeviceMesh") -> bool:
    """Check if mesh has unique device refs (true distributed vs simulated)."""
    return mesh is not None and len(set(mesh.device_refs)) == len(mesh.device_refs)

class ShardOp(Operation):
    """Split a replicated tensor into multiple sharded TensorValues.
    
    This operation takes a single TensorValue and produces a list of TensorValues,
    one per shard. Each shard contains a slice of the original tensor according
    to the sharding specification.
    
    Example:
        x: TensorValue shape (4, 8)
        spec: shard dim 0 on mesh axis "x" with 2 devices
        result: [TensorValue (2, 8), TensorValue (2, 8)]
    """
    
    @property
    def name(self) -> str:
        return "shard"
    
    def maxpr(
        self,
        x: TensorValue,
        mesh: DeviceMesh,
        dim_specs: List[DimSpec],
    ) -> List[TensorValue]:
        """Create sharded TensorValues by slicing the input.
        
        Args:
            x: Single TensorValue to shard
            mesh: Device mesh defining the shard topology
            dim_specs: Per-dimension sharding specification
            
        Returns:
            List of TensorValues, one per shard
        """
        from ..sharding.spec import ShardingSpec
        
        num_shards = len(mesh.devices)
        global_shape = tuple(int(d) for d in x.type.shape)
        spec = ShardingSpec(mesh, dim_specs)
        
        shard_values = []
        for shard_idx in range(num_shards):
            # Compute slice for this shard
            slices = self._compute_shard_slice(global_shape, spec, shard_idx)
            shard_val = x[slices]
            
            # Place on designated device if distributed
            if _is_distributed(mesh):
                shard_val = ops.transfer_to(shard_val, mesh.device_refs[shard_idx])
            
            shard_values.append(shard_val)
        
        return shard_values
    
    def _compute_shard_slice(
        self,
        global_shape: tuple,
        spec: ShardingSpec,
        shard_idx: int,
    ) -> tuple:
        """Compute slice indices for a specific shard."""
        slices = []
        
        for dim_idx, dim_len in enumerate(global_shape):
            if dim_idx >= len(spec.dim_specs):
                slices.append(slice(None))
                continue
            
            dim_spec = spec.dim_specs[dim_idx]
            
            if not dim_spec.axes:
                # Replicated - full slice
                slices.append(slice(None))
                continue
            
            # Compute shard position and size
            total_shards = 1
            my_shard_pos = 0
            
            for axis_name in dim_spec.axes:
                size = spec.mesh.get_axis_size(axis_name)
                coord = spec.mesh.get_coordinate(shard_idx, axis_name)
                my_shard_pos = (my_shard_pos * size) + coord
                total_shards *= size
            
            chunk_size = math.ceil(dim_len / total_shards)
            start = my_shard_pos * chunk_size
            
            # Ensure start is within bounds (MAX Graph slice op errors if start > length)
            # For empty shards at the end of uneven splits, start can be > dim_len
            start = min(start, dim_len)
            
            # End is always clamped to dim_len
            end = min(start + chunk_size, dim_len)
            slices.append(slice(start, end))
        
        return tuple(slices)
    
    def __call__(self, x, mesh: DeviceMesh, dim_specs: List[DimSpec]):
        """Shard a tensor according to the given specification.
        
        This overrides the base __call__ to handle multi-value output specially.
        """
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        from ..sharding.spec import ShardingSpec
        from max import graph as g
        
        # Store global shape BEFORE sharding (from input tensor)
        global_shape = None
        if isinstance(x, Tensor):
            # Get the global shape from input (which is still unsharded)
            global_shape = x._impl.cached_shape or (
                x._impl.physical_shape if x._impl.physical_shape else None
            )
        
        with GRAPH.graph:
            # Convert input to TensorValue
            x_val = g.TensorValue(x) if isinstance(x, Tensor) else x
            
            # Execute shard operation
            shard_values = self.maxpr(x_val, mesh, dim_specs)
        
        # Create output tensor with multiple values
        spec = ShardingSpec(mesh, dim_specs)
        impl = TensorImpl(
            values=shard_values,
            traced=x._impl.traced if isinstance(x, Tensor) else False,
            batch_dims=x._impl.batch_dims if isinstance(x, Tensor) else 0,
            sharding=spec,
        )
        
        # Cache GLOBAL shape from input, not local shard shape
        # This is critical for sharding propagation to work correctly
        if global_shape is not None:
            impl.cached_shape = global_shape
        elif shard_values:
            # Fallback: compute global from local shard shape
            local_shape = shard_values[0].type.shape
            impl.cached_shape = self._compute_global_from_local(local_shape, spec)
        
        # Cache dtype/device from first shard
        if shard_values:
            tensor_type = shard_values[0].type.as_tensor() if hasattr(shard_values[0].type, 'as_tensor') else shard_values[0].type
            impl.cached_dtype = tensor_type.dtype
            device = tensor_type.device
            impl.cached_device = device.to_device() if hasattr(device, 'to_device') else device
        
        return Tensor(impl=impl)
    
    def _compute_global_from_local(self, local_shape, sharding):
        """Compute global shape from local shard shape and sharding spec."""
        from max import graph
        
        global_dims = []
        for dim_idx, dim in enumerate(local_shape):
            dim_val = int(dim)
            if dim_idx < len(sharding.dim_specs):
                dim_spec = sharding.dim_specs[dim_idx]
                if dim_spec.axes:
                    shard_count = dim_spec.get_total_shards(sharding.mesh)
                    dim_val = dim_val * shard_count
            global_dims.append(dim_val)
        
        return graph.Shape(global_dims)


class AllGatherOp(Operation):
    """Gather shards along an axis to produce replicated full tensors.
    
    Takes N sharded TensorValues and produces N replicated TensorValues,
    each containing the full concatenated tensor.
    """
    
    @property
    def name(self) -> str:
        return "all_gather"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        axis: int,
        mesh: "DeviceMesh" = None,
        sharded_axis_name: str = None,
    ) -> List[TensorValue]:
        """Gather all shards along the specified axis.
        
        Args:
            shard_values: List of shard TensorValues
            axis: Axis along which shards are split
            mesh: Device mesh (needed for 2D+ meshes to identify unique shards)
            sharded_axis_name: Name of the mesh axis that was sharded
            
        Returns:
            List of replicated TensorValues (all identical)
        """
        # DISTRIBUTED: Use native MAX allgather
        if _is_distributed(mesh):
            from max.graph.ops.allgather import allgather as max_allgather
            from max.graph.type import BufferType
            from max.dtype import DType
            
            # Create signal buffers (one per device)
            signal_buffers = [
                ops.buffer_create(BufferType(DType.int64, (1,), dev))
                for dev in mesh.device_refs
            ]
            return max_allgather(shard_values, signal_buffers, axis=axis)
        
        # SIMULATED: Concat-based fallback
        # For 2D+ meshes, we need to select only unique shards
        # (avoid concatenating duplicates from non-sharded mesh axes)
        if mesh is not None and sharded_axis_name is not None:
            sharded_axis_size = mesh.get_axis_size(sharded_axis_name)
            
            # Group shards by their coordinate on the sharded axis
            unique_shards = []
            seen_coords = set()
            
            for shard_idx, val in enumerate(shard_values):
                coord = mesh.get_coordinate(shard_idx, sharded_axis_name)
                if coord not in seen_coords:
                    seen_coords.add(coord)
                    unique_shards.append((coord, val))
            
            # Sort by coordinate and extract values
            unique_shards.sort(key=lambda x: x[0])
            shards_to_concat = [v for _, v in unique_shards]
        else:
            shards_to_concat = shard_values
        
        # Concatenate unique shards to get full tensor
        if len(shards_to_concat) == 1:
            full_tensor = shards_to_concat[0]
        else:
            full_tensor = ops.concat(shards_to_concat, axis=axis)
        
        # Return N copies (one per original shard location)
        return [full_tensor] * len(shard_values)
    
    def __call__(self, sharded_tensor, axis: int):
        """Gather all shards to produce a replicated tensor.
        
        Args:
            sharded_tensor: Tensor with multiple shards
            axis: Axis along which shards are split
            
        Returns:
            Replicated tensor with gathered values
        """
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        from ..sharding.spec import ShardingSpec, DimSpec
        
        if (not sharded_tensor._impl._values or len(sharded_tensor._impl._values) <= 1) and \
           (not sharded_tensor._impl.sharding or sharded_tensor._impl.sharding.is_fully_replicated()):
            return sharded_tensor  # Already logically replicated
            
        # Get mesh and sharded axis info from input sharding
        mesh = sharded_tensor._impl.sharding.mesh if sharded_tensor._impl.sharding else None
        sharded_axis_name = None
        
        if sharded_tensor._impl.sharding:
            # Find which mesh axis this tensor dimension is sharded on
            sharding = sharded_tensor._impl.sharding
            if axis < len(sharding.dim_specs) and sharding.dim_specs[axis].axes:
                sharded_axis_name = sharding.dim_specs[axis].axes[0]
        
        if not sharded_tensor._impl._values or len(sharded_tensor._impl._values) <= 1:
            # Physically gathered (single value) but logically sharded.
            # We just need to update the metadata to be replicated.
            rank = len(sharded_tensor.shape)
            replicated_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)]) if mesh else None
            
            impl = TensorImpl(
                storages=sharded_tensor._impl._storages,  # Copy storages!
                values=sharded_tensor._impl._values,
                traced=sharded_tensor._impl.traced,
                batch_dims=sharded_tensor._impl.batch_dims,
                sharding=replicated_spec,
            )
            # Ensure shape is propagated (force resolution if needed)
            impl.cached_shape = sharded_tensor.shape
            impl.cached_dtype = sharded_tensor.dtype
            impl.cached_device = sharded_tensor.device
            
            return Tensor(impl=impl)
        
        with GRAPH.graph:
            gathered = self.maxpr(
                sharded_tensor._impl._values, axis, 
                mesh=mesh, sharded_axis_name=sharded_axis_name
            )
        
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
        
        # Create output tensor
        impl = TensorImpl(
            values=gathered,
            traced=sharded_tensor._impl.traced,
            batch_dims=sharded_tensor._impl.batch_dims,
            sharding=output_spec,
        )
        return Tensor(impl=impl)


class AllReduceOp(Operation):
    """Reduce values across all shards using the specified reduction.
    
    Takes N TensorValues with partial results and produces N TensorValues
    with the fully reduced result (all identical after reduction).
    """
    
    @property
    def name(self) -> str:
        return "all_reduce"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        mesh: "DeviceMesh" = None,
    ) -> List[TensorValue]:
        """Sum-reduce across all shards (AllReduce).
        
        Note: MAX only supports sum reduction natively. Other reductions
        (mean, max, min) are not supported.
        
        Args:
            shard_values: List of shard TensorValues to reduce
            mesh: Device mesh (needed for distributed execution)
            
        Returns:
            List of reduced TensorValues (all identical)
        """
        if not shard_values:
            return []
        
        # DISTRIBUTED: Use native MAX allreduce
        if _is_distributed(mesh):
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
    
    def __call__(self, sharded_tensor):
        """Sum-reduce across all shards.
        
        Args:
            sharded_tensor: Tensor with partial values per shard
            
        Returns:
            Tensor with sum-reduced values (replicated across shards)
        """
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        from ..sharding.spec import ShardingSpec, DimSpec
        
        if not sharded_tensor._impl._values or len(sharded_tensor._impl._values) <= 1:
            return sharded_tensor  # Nothing to reduce
        
        # Get mesh from tensor's sharding spec
        mesh = sharded_tensor._impl.sharding.mesh if sharded_tensor._impl.sharding else None
        
        with GRAPH.graph:
            reduced = self.maxpr(sharded_tensor._impl._values, mesh=mesh)
        
        # Create replicated sharding spec (all dims have empty axes list)
        output_spec = None
        if mesh and reduced:
            rank = len(reduced[0].type.shape) if reduced else 0
            output_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])
        
        impl = TensorImpl(
            values=reduced,
            traced=sharded_tensor._impl.traced,
            batch_dims=sharded_tensor._impl.batch_dims,
            sharding=output_spec,
        )
        return Tensor(impl=impl)


class ReduceScatterOp(Operation):
    """Reduce then scatter the result across shards.
    
    Each shard receives a different portion of the reduced result.
    """
    
    @property
    def name(self) -> str:
        return "reduce_scatter"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        axis: int,
        mesh: "DeviceMesh" = None,
    ) -> List[TensorValue]:
        """Sum-reduce across shards then scatter the result.
        
        Note: MAX only supports sum reduction natively.
        
        Args:
            shard_values: List of shard TensorValues
            axis: Axis along which to scatter the reduced result
            mesh: Device mesh (needed for distributed execution)
            
        Returns:
            List of scattered TensorValues
        """
        # DISTRIBUTED: Compose allreduce + scatter
        if _is_distributed(mesh):
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
    
    def __call__(self, sharded_tensor, axis: int):
        """Sum-reduce then scatter across shards.
        
        Args:
            sharded_tensor: Tensor with values to reduce
            axis: Axis along which to scatter the result
            
        Returns:
            Tensor with scattered reduced values
        """
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        from ..sharding.spec import ShardingSpec, DimSpec
        
        if not sharded_tensor._impl._values or len(sharded_tensor._impl._values) <= 1:
            return sharded_tensor  # Nothing to reduce
        
        mesh = sharded_tensor._impl.sharding.mesh if sharded_tensor._impl.sharding else None
        
        with GRAPH.graph:
            scattered = self.maxpr(sharded_tensor._impl._values, axis, mesh=mesh)
        
        # Output sharding: the scatter axis becomes sharded
        output_spec = None
        if mesh and sharded_tensor._impl.sharding:
            input_spec = sharded_tensor._impl.sharding
            # Create new spec where the scatter axis is sharded
            new_dim_specs = []
            rank = len(scattered[0].type.shape) if scattered else 0
            for d in range(rank):
                if d < len(input_spec.dim_specs):
                    new_dim_specs.append(input_spec.dim_specs[d])
                else:
                    new_dim_specs.append(DimSpec([]))
            output_spec = ShardingSpec(mesh, new_dim_specs)
        
        impl = TensorImpl(
            values=scattered,
            traced=sharded_tensor._impl.traced,
            batch_dims=sharded_tensor._impl.batch_dims,
            sharding=output_spec,
        )
        return Tensor(impl=impl)


# Singleton instances
shard_op = ShardOp()
all_gather_op = AllGatherOp()
all_reduce_op = AllReduceOp()
reduce_scatter_op = ReduceScatterOp()


# Public API functions
def shard(x, mesh: DeviceMesh, dim_specs: List[DimSpec]):
    """Shard a tensor according to the given mesh and dimension specs.
    
    Args:
        x: Input tensor (replicated/unsharded)
        mesh: Device mesh defining shard topology
        dim_specs: List of DimSpec for each dimension
        
    Returns:
        Sharded tensor with multiple internal TensorValues
    """
    return shard_op(x, mesh, dim_specs)


def all_gather(sharded_tensor, axis: int):
    """Gather all shards to produce a replicated tensor.
    
    Args:
        sharded_tensor: Tensor with multiple shards
        axis: Axis along which shards are split
        
    Returns:
        Replicated tensor with gathered values
    """
    return all_gather_op(sharded_tensor, axis)


def all_reduce(sharded_tensor):
    """Sum-reduce across all shards.
    
    Note: MAX only supports sum reduction natively.
    
    Args:
        sharded_tensor: Tensor with partial values per shard
        
    Returns:
        Tensor with sum-reduced values (replicated across shards)
    """
    return all_reduce_op(sharded_tensor)


def reduce_scatter(sharded_tensor, axis: int):
    """Sum-reduce then scatter result across shards.
    
    Note: MAX only supports sum reduction natively.
    
    Args:
        sharded_tensor: Tensor with values per shard
        axis: Axis along which to scatter
        
    Returns:
        Tensor with scattered reduced values
    """
    return reduce_scatter_op(sharded_tensor, axis)





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
                
        # After processing all sharded axes, we should have 1 result (replicated)
        # Or multiple if some axes were unused (replicated).
        # If replicated axes exist in mesh, `current_shard_descs` has N copies.
        # We just pick the first one.
        
        return current_shard_descs[0][0]
    
    def __call__(self, sharded_tensor):
        """Gather all sharded axes to produce a replicated tensor.
        
        Args:
            sharded_tensor: Tensor with any sharding configuration
            
        Returns:
            Replicated tensor with the full global data
        """
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        from ..sharding.spec import ShardingSpec, DimSpec
        from max import graph as g
        
        if not sharded_tensor._impl.sharding:
            return sharded_tensor  # No sharding info
        
        spec = sharded_tensor._impl.sharding
        mesh = spec.mesh
        
        if spec.is_fully_replicated():
            return sharded_tensor  # Already replicated
        
        # Access TensorValues - this triggers add_input for realized tensors
        # via __tensorvalue__ if needed
        shard_values = sharded_tensor._impl._values
        
        # If realized but _values is empty, the tensor needs to be added as input
        if not shard_values and sharded_tensor._impl._storages:
            GRAPH.add_input(sharded_tensor)
            shard_values = sharded_tensor._impl._values
        
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
            sharding=replicated_spec,
        )
        return Tensor(impl=impl)


# Singleton for GatherAllAxesOp (defined before ReshardOp for ordering)
gather_all_axes_op = GatherAllAxesOp()


def gather_all_axes(sharded_tensor):
    """Gather all sharded axes to produce a fully replicated tensor.
    
    Args:
        sharded_tensor: Tensor with any sharding configuration
        
    Returns:
        Replicated tensor with the full global data
    """
    return gather_all_axes_op(sharded_tensor)


class ReshardOp(Operation):
    """Generic resharding between different specifications.
    
    This operation handles complex transitions like:
    - Axis permutation (Transpose sharding)
    - Splitting/Merging axes
    - Changing device meshes (if device list same)
    
    Implementation Strategy:
    1. Reconstruct Global Tensor (Gather all shards)
    2. Slice Global Tensor according to new spec (Scatter)
    
    Note: This is a memory-intensive simulation. Real AllToAll would exchange
    only necessary chunks.
    """
    
    @property
    def name(self) -> str:
        return "reshard"
    


    def maxpr(
        self,
        shard_values: List[TensorValue],
        source_spec: "ShardingSpec",
        target_spec: "ShardingSpec"
    ) -> List[TensorValue]:
        """Execute resharding logic."""
        # 1. Reconstruct logical global tensor
        global_tensor = gather_all_axes_op.maxpr(shard_values, source_spec)
        
        # 2. Shard using target spec
        # Reuse ShardOp logic
        return shard_op.maxpr(global_tensor, target_spec.mesh, target_spec.dim_specs)

    def __call__(self, tensor, target_spec: "ShardingSpec"):
        """Reshard tensor to target spec."""
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        
        if not tensor._impl.sharding:
            # Assume tensor is replicated input, just shard it
            return shard_op(tensor, target_spec.mesh, target_spec.dim_specs)
            
        with GRAPH.graph:
            vals = self.maxpr(tensor._impl._values, tensor._impl.sharding, target_spec)
            
        impl = TensorImpl(
            values=vals,
            sharding=target_spec,
            traced=tensor._impl.traced,
            batch_dims=tensor._impl.batch_dims
        )
        # Propagate cached shape/dtype
        impl.cached_shape = tensor.shape
        impl.cached_dtype = tensor.dtype
        impl.cached_device = tensor.device
        
        return Tensor(impl=impl)

reshard_op = ReshardOp()

def reshard(tensor, target_spec):
    """Reshard a tensor to a new specification.
    
    Args:
        tensor: Input sharded tensor
        target_spec: Target ShardingSpec
        
    Returns:
        New tensor sharded according to target_spec
    """
    return reshard_op(tensor, target_spec)


def simulate_grouped_all_reduce(
    shard_results: List[TensorValue], 
    mesh: "DeviceMesh", 
    reduce_axes: "Set[str]",
    all_reduce_op: AllReduceOp
) -> List[TensorValue]:
    """Simulate grouped AllReduce execution for SPMD verification.
    
    When only a subset of mesh axes are involved in reduction (e.g., partial results
    from tensor parallelism on 'model' axis), we must effectively:
    1. Group shards that share coordinates on non-reduced axes
    2. AllReduce within each group
    3. Broadcast the result to all members of the group
    
    Args:
        shard_results: List of shard values (length = mesh size)
        mesh: The device mesh
        reduce_axes: Set of mesh axis names to reduce over
        all_reduce_op: The AllReduce operator to use
        
    Returns:
        List of reduced shard values (same length, grouped values are identical)
    """
    if not reduce_axes:
        return shard_results
        
    num_shards = len(shard_results)
    
    # Check if we're reducing over all axes (simple case)
    all_axes = set(mesh.axis_names)
    if all_axes.issubset(reduce_axes):
        return all_reduce_op.maxpr(shard_results, mesh=mesh)
        
    # Complex case: Group shards by non-reduced axes
    # Each group contains shards that should be reduced together
    groups = {}
    
    for shard_idx, result in enumerate(shard_results):
        # Build key from coords on NON-reduced axes
        key_parts = []
        for axis_name in mesh.axis_names:
            if axis_name not in reduce_axes:
                key_parts.append(mesh.get_coordinate(shard_idx, axis_name))
        
        # Use tuple of coords as grouping key
        key = tuple(key_parts)
        if key not in groups:
            groups[key] = []
        groups[key].append((shard_idx, result))
    
    # Execute reduction per group
    new_results = [None] * num_shards
    
    for key, group_members in groups.items():
        # group_members is a list of (shard_idx, shard_value)
        group_shards = [val for _, val in group_members]
        
        if len(group_shards) > 1:
            curr_reduced = all_reduce_op.maxpr(group_shards, mesh=mesh)
        else:
            curr_reduced = [group_shards[0]]
            
        # Distribute results back to original positions
        for i, (shard_idx, _) in enumerate(group_members):
            new_results[shard_idx] = curr_reduced[i] if isinstance(curr_reduced, list) else curr_reduced
            
    return new_results


__all__ = [
    # Op classes
    "ShardOp",
    "AllGatherOp", 
    "AllReduceOp",
    "ReduceScatterOp",
    "GatherAllAxesOp",
    "ReshardOp",
    # Singletons
    "shard_op",
    "all_gather_op",
    "all_reduce_op", 
    "reduce_scatter_op",
    "gather_all_axes_op",
    "reshard_op",
    # Public functions
    "shard",
    "all_gather",
    "all_reduce",
    "reduce_scatter",
    "gather_all_axes",
    "reshard",
    "simulate_grouped_all_reduce",
]
