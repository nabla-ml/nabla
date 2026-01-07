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


class CollectiveOperation(Operation):
    """Base class for collective communication operations.
    
    Reduces boilerplate for operations that:
    1. Take a sharded/replicated tensor as input
    2. Hydrate its values
    3. Perform a MAX graph operation (maxpr)
    4. Return a new Tensor with updated sharding spec
    """
    
    def __call__(self, sharded_tensor, **kwargs):
        from ..core.tensor import Tensor
        from ..core.tensor_impl import TensorImpl
        from ..core.compute_graph import GRAPH
        
        # 1. Validation and early exit
        if not self._should_proceed(sharded_tensor):
            return sharded_tensor
            
        mesh = sharded_tensor._impl.sharding.mesh if sharded_tensor._impl.sharding else None
        
        # 2. Execution in graph context
        with GRAPH.graph:
            from ..sharding.spmd import ensure_shard_values
            ensure_shard_values(sharded_tensor)
            
            # Call the specific implementation
            result_values = self.maxpr(sharded_tensor._impl._values, mesh=mesh, **kwargs)
            
        # 3. Output wrapping
        output_spec = self._compute_output_spec(sharded_tensor, result_values, **kwargs)
        
        impl = TensorImpl(
            values=result_values,
            traced=sharded_tensor._impl.traced,
            batch_dims=sharded_tensor._impl.batch_dims,
            sharding=output_spec,
        )
        output = Tensor(impl=impl)
        
        # 4. Tracing setup
        self._setup_output_refs(output, (sharded_tensor,), kwargs, sharded_tensor._impl.traced)
        
        return output
        
    def _should_proceed(self, tensor):
        """Check if operation should proceed (values exist, not trivial)."""
        # Default: proceed if we have values > 1
        return tensor._impl._values and len(tensor._impl._values) > 1
        
    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Compute output sharding spec. Default: preserve input spec."""
        return input_tensor._impl.sharding

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
    
    def infer_sharding_spec(self, args, mesh, kwargs):
        spec = kwargs['spec']
        # Return input sharding to allow partial reuse/identity
        input_spec = args[0]._impl.sharding
        return spec, [input_spec], False
    
    def maxpr(
        self,
        x: TensorValue,
        mesh: DeviceMesh,
        dim_specs: List[DimSpec],
        **kwargs: Any,
    ) -> List[TensorValue]:
        """Create sharded TensorValues by slicing the input.
        
        Args:
            x: Single TensorValue to shard
            mesh: Device mesh defining the shard topology
            dim_specs: Per-dimension sharding specification
            kwargs: Must include 'shard_idx' or we are in manual call mode?
                    If manual call (e.g. simulation loop in ShardOp.__call__), we loop over shards.
        """
        from ..sharding.spec import ShardingSpec, compute_local_shape
        
        # Determine execution context
        # If kwargs contains shard_idx, we are in SPMD execution (called by Operation.__call__)
        # If not, we are in ShardOp.__call__ (manual loop) or similar.
        
        global_shape = tuple(int(d) for d in x.type.shape)
        # Note: If x is sharded, default type.shape is local shape!
        # But ShardOp inputs are sometimes global (in simulation) or local.
        # We need to rely on what Operation.__call__ passes.
        # In SPMD mode, args[0] is the local shard of input.
        
        spec = ShardingSpec(mesh, dim_specs)
        
        # If operating in SPMD mode
        if "shard_idx" in kwargs:
             shard_idx = kwargs["shard_idx"]
             return self._slice_for_device(x, global_shape, spec, shard_idx, mesh)
        
        # Manual loop mode (simulated execution called from ShardOp.__call__)
        num_shards = len(mesh.devices)
        shard_values = []
        for shard_idx in range(num_shards):
            val = self._slice_for_device(x, global_shape, spec, shard_idx, mesh)
            
            if _is_distributed(mesh):
                val = ops.transfer_to(val, mesh.device_refs[shard_idx])
            shard_values.append(val)
            
        return shard_values

    def _slice_for_device(self, x, global_shape, spec, shard_idx, mesh):
        from ..sharding.spec import compute_local_shape
        
        # Target local shape for this device
        target_local_shape = compute_local_shape(global_shape, spec, shard_idx)
        
        slices = []
        for d, (t_len, g_len) in enumerate(zip(target_local_shape, global_shape)):
            # Check input dimension length
            inp_len = int(x.type.shape[d])
            
            if inp_len == t_len:
                # Identity slice (input matches target)
                slices.append(slice(0, t_len))
                continue
            
            # Must compute offset in global coords
            # (Re-implementing logic from compute_local_shape to get offset)
            dim_spec = spec.dim_specs[d]
            total_shards = 1
            my_shard_pos = 0
            for axis_name in dim_spec.axes:
                size = mesh.get_axis_size(axis_name)
                coord = mesh.get_coordinate(shard_idx, axis_name)
                my_shard_pos = (my_shard_pos * size) + coord
                total_shards *= size
            
            chunk_size = math.ceil(g_len / total_shards)
            start = my_shard_pos * chunk_size
            start = min(start, g_len)
            
            # If input is GLOBAL length, slice from start
            if inp_len == g_len:
                # Normal slice
                end = min(start + chunk_size, g_len)
                slices.append(slice(start, end))
            else:
                 # Input length mismatch?
                 # If input is not target length AND not global length, we might have an issue.
                 # E.g. input is sharded differently.
                 # But infer_sharding_spec only allowed reuse if input_spec was compatible.
                 # For now, assume it must be global if not matching target.
                 # Or check robustness.
                 # If we are here, we fallback to slicing assuming input is global?
                 # This might fail if input is partial.
                 end = min(start + chunk_size, g_len)
                 slices.append(slice(start, end))
                 
        return x[tuple(slices)]

    def __call__(self, x, mesh: DeviceMesh, dim_specs: List[DimSpec], replicated_axes: Optional[Set[str]] = None):
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
            # Get the global physical shape from input
            # ShardOp must operate on physical shape to handle batch_dims correctly
            global_shape = x._impl.physical_shape if x._impl.physical_shape else x.global_shape
        
        with GRAPH.graph:
            # Convert input to TensorValue
            x_val = g.TensorValue(x) if isinstance(x, Tensor) else x
            
            # Execute shard operation
            shard_values = self.maxpr(x_val, mesh, dim_specs)
        
        # Create output tensor with multiple values
        spec = ShardingSpec(mesh, dim_specs, replicated_axes=replicated_axes or set())
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
        
        output = Tensor(impl=impl)
        
        # Setup tracing refs for graph traversal
        traced = x._impl.traced if isinstance(x, Tensor) else False
        self._setup_output_refs(output, (x,), {'mesh': mesh, 'dim_specs': dim_specs}, traced)
        
        return output
    
    
    def _compute_global_from_local(self, local_shape, sharding):
        """Deprecated: use spmd.compute_global_shape."""
        from ..sharding.spmd import compute_global_shape
        return compute_global_shape(local_shape, sharding)



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
            List of TensorValues - gathered along the specified axis.
            For 2D+ meshes, each device gets its appropriate slice based on
            coordinates on OTHER (non-gathered) mesh axes.
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
        
        # SIMULATED: For multi-dimensional meshes, we need to gather within
        # groups that share the same coordinates on non-gathered axes.
        #
        # Example: 2x2 mesh with axes ("x", "y"), gathering along "x":
        #   Group 1 (y=0): devices 0,1 -> concat their data
        #   Group 2 (y=1): devices 2,3 -> concat their data  
        #   Device 0 gets Group 1 result, Device 2 gets Group 2 result, etc.
        
        if mesh is None or sharded_axis_name is None:
            # Fallback: simple concat all
            if len(shard_values) == 1:
                return shard_values
            full_tensor = ops.concat(shard_values, axis=axis)
            return [full_tensor] * len(shard_values)
        
        # Get all mesh axis names
        all_axes = list(mesh.axis_names)
        other_axes = [ax for ax in all_axes if ax != sharded_axis_name]
        
        if not other_axes:
            # 1D mesh: simple case - gather all, return same to all
            sharded_axis_size = mesh.get_axis_size(sharded_axis_name)
            
            # Group by coordinate on the sharded axis
            unique_shards = []
            seen_coords = set()
            for shard_idx, val in enumerate(shard_values):
                coord = mesh.get_coordinate(shard_idx, sharded_axis_name)
                if coord not in seen_coords:
                    seen_coords.add(coord)
                    unique_shards.append((coord, val))
            unique_shards.sort(key=lambda x: x[0])
            shards_to_concat = [v for _, v in unique_shards]
            
            if len(shards_to_concat) == 1:
                full_tensor = shards_to_concat[0]
            else:
                full_tensor = ops.concat(shards_to_concat, axis=axis)
            
            return [full_tensor] * len(shard_values)
        
        # Multi-dimensional mesh: group devices by their coordinates on OTHER axes
        # Each group will produce its own gathered result
        
        # Build groups: key = tuple of coordinates on other axes
        groups = {}  # {other_coords: [(sharded_coord, shard_idx, value)]}
        for shard_idx, val in enumerate(shard_values):
            # Get this device's coordinates on OTHER axes
            other_coords = tuple(mesh.get_coordinate(shard_idx, ax) for ax in other_axes)
            # Get coordinate on the axis being gathered
            sharded_coord = mesh.get_coordinate(shard_idx, sharded_axis_name)
            
            if other_coords not in groups:
                groups[other_coords] = []
            groups[other_coords].append((sharded_coord, shard_idx, val))
        
        # For each group, gather along the sharded axis
        gathered_per_group = {}  # {other_coords: gathered_tensor}
        for other_coords, members in groups.items():
            # Sort by coordinate on the sharded axis
            members.sort(key=lambda x: x[0])
            shards_to_concat = [val for _, _, val in members]
            
            if len(shards_to_concat) == 1:
                gathered = shards_to_concat[0]
            else:
                gathered = ops.concat(shards_to_concat, axis=axis)
            
            gathered_per_group[other_coords] = gathered
        
        # Assign each device its group's gathered result
        results = []
        for shard_idx in range(len(shard_values)):
            other_coords = tuple(mesh.get_coordinate(shard_idx, ax) for ax in other_axes)
            results.append(gathered_per_group[other_coords])
        
        return results
    
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
        
        # Ensure values exist (hydrate from storage if needed)
        with GRAPH.graph:
            from ..sharding.spmd import ensure_shard_values
            ensure_shard_values(sharded_tensor)
            
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
        output = Tensor(impl=impl)
        
        # Setup tracing refs for graph traversal
        self._setup_output_refs(output, (sharded_tensor,), {'axis': axis}, sharded_tensor._impl.traced)
        
        return output


class AllReduceOp(CollectiveOperation):
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
    
    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output is fully replicated."""
        from ..sharding.spec import ShardingSpec, DimSpec
        mesh = input_tensor._impl.sharding.mesh if input_tensor._impl.sharding else None
        
        if mesh and results:
            rank = len(results[0].type.shape)
            return ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])
        return None


class ReduceScatterOp(CollectiveOperation):
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
    
    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output sharding: the scatter axis becomes sharded."""
        from ..sharding.spec import ShardingSpec, DimSpec
        
        mesh = input_tensor._impl.sharding.mesh if input_tensor._impl.sharding else None
        input_spec = input_tensor._impl.sharding
        
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

class ReshardOp(Operation):
    """Generic resharding operation.
    
    Reshards a tensor from its current sharding (or replication) to a new target
    sharding specification. Handles both logical to physical spec conversion
    (for batch_dims) and the actual data movement (gather + shard).
    """
    
    @property
    def name(self) -> str:
        return "reshard"
    
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
        from ..sharding.spec import ShardingSpec, DimSpec, needs_reshard
        from ..core.tensor import Tensor
        
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
            # Just return input (maybe update spec if None -> Replicated implicit)
            # Or assume caller handles metadata update? 
            # Ideally return tensor as-is if strictly no change.
            # But ensure spec is set if it was None?
            if current_spec is None:
                 tensor._impl.sharding = target_spec
            return tensor

        # 4. Perform Resharding (Gather + Shard)
        # Gather all sharded axes to get global data
        result = tensor
        if current_spec:
            from . import all_gather
            for dim, dim_spec in enumerate(current_spec.dim_specs):
                if dim_spec.axes:
                    result = all_gather(result, axis=dim)
        
        # Shard to target using module-level shard_op (efficient)
        result = shard_op(result, mesh, target_spec.dim_specs, replicated_axes=target_spec.replicated_axes)
        
        return result



class PPermuteOp(CollectiveOperation):
    """Point-to-point permutation collective.
    
    Each device sends its value to exactly one other device according to a
    permutation table. This is useful for ring-based algorithms, pipeline
    parallelism, and halo exchange.
    
    Example:
        # Ring shift: device 0→1→2→3→0
        perm = [(0, 1), (1, 2), (2, 3), (3, 0)]
        y = ppermute(x, perm)
    """
    
    @property
    def name(self) -> str:
        return "ppermute"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        permutation: List[tuple],
        mesh: "DeviceMesh" = None,
    ) -> List[TensorValue]:
        """Permute values between devices according to permutation.
        
        Args:
            shard_values: List of TensorValues, one per device
            permutation: List of (source_idx, dest_idx) pairs
            mesh: Device mesh for distributed execution
            
        Returns:
            List of TensorValues after permutation (zeros for missing dests)
        """
        num_devices = len(shard_values)
        
        # Build reverse map: dest -> src
        dest_to_src = {}
        for src, dst in permutation:
            if dst in dest_to_src:
                raise ValueError(f"Destination {dst} appears multiple times in permutation")
            dest_to_src[dst] = src
        
        # Create result array
        results = []
        
        for dst in range(num_devices):
            if dst in dest_to_src:
                src = dest_to_src[dst]
                val = shard_values[src]
                
                # DISTRIBUTED: Transfer to destination device
                if _is_distributed(mesh):
                    val = ops.transfer_to(val, mesh.device_refs[dst])
                
                results.append(val)
            else:
                # No sender for this destination - return zeros
                template = shard_values[0]
                zero_val = ops.constant(0, template.type.dtype, template.type.device)
                zero_val = ops.broadcast_to(zero_val, template.type.shape)
                results.append(zero_val)
        
        return results
    



class AllToAllOp(CollectiveOperation):
    """All-to-all collective (distributed transpose).
    
    Each device splits its tensor along split_axis, sends parts to other devices,
    receives from all, and concatenates along concat_axis. This is useful for
    expert routing (MoE), axis swapping, and distributed FFT.
    
    Example:
        # 4 devices, each has [4, N], after all_to_all each has [4, N] but
        # with data transposed across devices
        y = all_to_all(x, split_axis=0, concat_axis=0)
    """
    
    @property
    def name(self) -> str:
        return "all_to_all"
    
    def maxpr(
        self,
        shard_values: List[TensorValue],
        split_axis: int,
        concat_axis: int,
        mesh: "DeviceMesh" = None,
        tiled: bool = True,
    ) -> List[TensorValue]:
        """All-to-all: distributed transpose of tensor blocks.
        
        Args:
            shard_values: List of TensorValues, one per device
            split_axis: Axis along which to split each shard
            concat_axis: Axis along which to concatenate received chunks
            mesh: Device mesh for distributed execution
            tiled: If True, concatenate; if False, stack (adds new dim)
            
        Returns:
            List of TensorValues after all-to-all exchange
        """
        num_devices = len(shard_values)
        
        if num_devices <= 1:
            return shard_values
        
        # 1. Each device splits its tensor into num_devices chunks
        chunks_per_device = []
        for val in shard_values:
            shape = val.type.shape
            axis_size = int(shape[split_axis])
            chunk_size = axis_size // num_devices
            
            if axis_size % num_devices != 0:
                raise ValueError(
                    f"Split axis size {axis_size} not divisible by {num_devices} devices"
                )
            
            chunks = []
            for i in range(num_devices):
                slices = [slice(None)] * len(shape)
                slices[split_axis] = slice(i * chunk_size, (i + 1) * chunk_size)
                chunks.append(val[tuple(slices)])
            
            chunks_per_device.append(chunks)
        
        # 2. Transpose: device j collects chunk[i][j] from each device i
        received_per_device = []
        for dst in range(num_devices):
            received = []
            for src in range(num_devices):
                chunk = chunks_per_device[src][dst]
                
                # DISTRIBUTED: Transfer chunk to destination
                if _is_distributed(mesh):
                    chunk = ops.transfer_to(chunk, mesh.device_refs[dst])
                
                received.append(chunk)
            received_per_device.append(received)
        
        # 3. Each device concatenates (or stacks) received chunks
        results = []
        for dst in range(num_devices):
            if tiled:
                concatenated = ops.concat(received_per_device[dst], axis=concat_axis)
            else:
                concatenated = ops.stack(received_per_device[dst], axis=concat_axis)
            results.append(concatenated)
        
        return results
    



class AxisIndexOp(Operation):
    """Return the device's position along a mesh axis.
    
    This is essential for shard_map-style programming where each device
    needs to know its position for conditional logic.
    
    Example:
        idx = axis_index('i')  # Returns 0, 1, 2, 3 on 4 devices
    """
    
    @property
    def name(self) -> str:
        return "axis_index"
    
    def maxpr(
        self,
        mesh: "DeviceMesh",
        axis_name: str,
        shard_idx: int,
    ) -> TensorValue:
        """Return this device's index along the specified axis.
        
        Args:
            mesh: Device mesh
            axis_name: Name of axis to get index for
            shard_idx: Current shard index
            
        Returns:
            Scalar TensorValue with the axis index
        """
        coord = mesh.get_coordinate(shard_idx, axis_name)
        return ops.constant(coord, mesh.device_refs[shard_idx].dtype if hasattr(mesh.device_refs[shard_idx], 'dtype') else None)
    
    def __call__(self, mesh: "DeviceMesh", axis_name: str) -> List[TensorValue]:
        """Get axis indices for all devices.
        
        Args:
            mesh: Device mesh
            axis_name: Name of axis to get indices for
            
        Returns:
            List of scalar TensorValues, one per device
        """
        from ..core.compute_graph import GRAPH
        from max.dtype import DType
        from max.graph import DeviceRef
        
        results = []
        with GRAPH.graph:
            for shard_idx in range(len(mesh.devices)):
                coord = mesh.get_coordinate(shard_idx, axis_name)
                # Use device ref from mesh, or default to CPU
                device = mesh.device_refs[shard_idx] if mesh.device_refs else DeviceRef.CPU()
                val = ops.constant(coord, DType.int32, device)
                results.append(val)
        
        return results


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
        from ..sharding.spec import ShardingSpec, DimSpec
        mesh = input_tensor._impl.sharding.mesh if input_tensor._impl.sharding else None
        
        if mesh and results:
            rank = len(results[0].type.shape)
            return ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])
        return None


# Singleton instances
shard_op = ShardOp()
all_gather_op = AllGatherOp()
all_reduce_op = AllReduceOp()
reduce_scatter_op = ReduceScatterOp()
ppermute_op = PPermuteOp()
all_to_all_op = AllToAllOp()
axis_index_op = AxisIndexOp()
pmean_op = PMeanOp()


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
    return reduce_scatter_op(sharded_tensor, axis=axis)


def ppermute(sharded_tensor, permutation: List[tuple]):
    """Point-to-point permutation collective.
    
    Each device sends its value to exactly one other device according to
    a permutation table. Useful for ring-based algorithms, pipeline
    parallelism, and halo exchange.
    
    Args:
        sharded_tensor: Tensor with multiple shards
        permutation: List of (source_idx, dest_idx) pairs specifying
                     which device sends to which. Destinations without
                     senders receive zeros.
        
    Returns:
        Tensor with permuted values
        
    Example:
        # Ring shift: device 0→1→2→3→0
        perm = [(0, 1), (1, 2), (2, 3), (3, 0)]
        y = ppermute(x, perm)
    """
    return ppermute_op(sharded_tensor, permutation=permutation)


def all_to_all(sharded_tensor, split_axis: int, concat_axis: int, tiled: bool = True):
    """All-to-all collective (distributed transpose).
    
    Each device splits its tensor along split_axis, sends parts to other
    devices, receives from all, and concatenates along concat_axis. Useful
    for expert routing (MoE), axis swapping, and distributed FFT.
    
    Note: Currently simulated using transfer_to. Will use native MAX
    all_to_all when available for better performance.
    
    Args:
        sharded_tensor: Tensor with multiple shards
        split_axis: Axis along which to split each shard
        concat_axis: Axis along which to concatenate received chunks
        tiled: If True, concatenate; if False, stack (adds new dimension)
        
    Returns:
        Tensor after all-to-all exchange
        
    Example:
        # 4 devices, each has [4, 8], exchange along first axis
        y = all_to_all(x, split_axis=0, concat_axis=0)
    """
    return all_to_all_op(sharded_tensor, split_axis=split_axis, concat_axis=concat_axis, tiled=tiled)


def axis_index(mesh: "DeviceMesh", axis_name: str) -> List[TensorValue]:
    """Return each device's position along a mesh axis.
    
    Essential for shard_map-style programming where devices need to
    know their position for conditional logic.
    
    Args:
        mesh: Device mesh
        axis_name: Name of axis to get indices for
        
    Returns:
        List of scalar TensorValues, one per device, containing 0, 1, 2, ...
        
    Example:
        # 4 devices along axis 'i'
        indices = axis_index(mesh, 'i')  # [0, 1, 2, 3]
    """
    return axis_index_op(mesh, axis_name)


def pmean(sharded_tensor, axis_name: str = None):
    """Compute mean across all shards.
    
    Equivalent to psum(x) / axis_size. Useful for averaging gradients
    in data parallelism.
    
    Args:
        sharded_tensor: Tensor with partial values per shard
        axis_name: Name of mesh axis for size calculation (optional)
        
    Returns:
        Tensor with mean values (replicated across shards)
        
    Example:
        # Average gradients across data-parallel workers
        avg_grad = pmean(grad_shard, 'dp')
    """
    return pmean_op(sharded_tensor, axis_name=axis_name)


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
    "PPermuteOp",
    "AllToAllOp",
    "AxisIndexOp",
    "PMeanOp",
    "GatherAllAxesOp",
    "ReshardOp",
    # Singletons
    "shard_op",
    "all_gather_op",
    "all_reduce_op", 
    "reduce_scatter_op",
    "ppermute_op",
    "all_to_all_op",
    "axis_index_op",
    "pmean_op",
    "gather_all_axes_op",
    "reshard_op",
    # Public functions
    "shard",
    "shard_batch_dims",
    "all_gather",
    "all_reduce",
    "reduce_scatter",
    "ppermute",
    "all_to_all",
    "axis_index",
    "pmean",
    "pmean",
    "gather_all_axes",
    "reshard",
    "simulate_grouped_all_reduce",
]


def shard_batch_dims(
    tensor: "Tensor", 
    mesh: "DeviceMesh", 
    axis_names: str | List[str], 
    batch_axis: int = 0
) -> "Tensor":
    """Shard the batch dimension(s) of a tensor (e.g. inside vmap).
    
    This allows explicit control over how vmapped batch dimensions map to the device mesh,
    enabling JAX-like `spmd_axis_name` functionality.
    
    Args:
        tensor: The tensor to shard (must have batch_dims > 0)
        mesh: The device mesh
        axis_names: Mesh axis name(s) to shard the batch dimension(s) on.
                    Can be a single string (shard outer batch dim) or list (shard multiple).
        batch_axis: Which batch dimension to start sharding from (relative to batch dims).
                    Default 0 (the outermost batch dimension).
                    
    Returns:
        Tensor sharded on the batch dimension(s).
    """
    from ..sharding.spec import ShardingSpec, DimSpec
    from ..sharding.spmd import reshard_tensor
    
    if tensor.batch_dims == 0:
        raise ValueError("Cannot shard batch dims of a tensor with no batch dimensions (not inside vmap?)")

    if isinstance(axis_names, str):
        axis_names = [axis_names]
        
    if batch_axis + len(axis_names) > tensor.batch_dims:
        raise ValueError(
            f"Cannot shard {len(axis_names)} axes starting at batch_axis {batch_axis}: "
            f"tensor only has {tensor.batch_dims} batch dimensions."
        )

    # Start with existing spec or create fully replicated one
    current_spec = tensor._impl.sharding
    physical_rank = len(tensor._impl.physical_shape) if tensor._impl.physical_shape else (len(tensor.shape) + tensor.batch_dims)
    
    if current_spec:
        # Clone existing specs
        new_specs = [ds.clone() for ds in current_spec.dim_specs]
        # Pad if missing (shouldn't happen with new robust logic but be safe)
        while len(new_specs) < physical_rank:
            new_specs.append(DimSpec([]))
    else:
        # Create fully replicated specs
        new_specs = [DimSpec([]) for _ in range(physical_rank)]
        
    # Update the specific batch dimensions
    for i, axis_name in enumerate(axis_names):
        assert axis_name in mesh.axis_names, f"Axis {axis_name} not in mesh {mesh.axis_names}"
        
        # Physical index of the batch dimension
        # batch dims are always at the start (0..batch_dims-1)
        idx = batch_axis + i
        new_specs[idx] = DimSpec([axis_name], is_open=True) # Open? Or Closed?
        # Usually internal vmap batching is "open" conceptually, but DimSpec is data layout.
        
    target_spec = ShardingSpec(mesh, new_specs)
    
    # Reshard
    # Use generic ReshardOp
    return reshard(tensor, mesh, new_specs)

reshard = ReshardOp()
shard = shard_op

__all__ = [
    "CollectiveOperation",
    "ShardOp", "AllGatherOp", "AllReduceOp", "ReduceScatterOp", "PPermuteOp", "AllToAllOp", "AxisIndexOp", "ReshardOp",
    "shard", "all_gather", "all_reduce", "reduce_scatter", "ppermute", "all_to_all", "axis_index", "reshard",
    "shard_batch_dims",
]
