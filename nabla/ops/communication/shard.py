# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional, Set

from max.graph import TensorValue, ops

from ..base import Operation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
    from ...core import Tensor


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
        input_spec = args[0].sharding
        return spec, [input_spec], False
    
    def maxpr(
        self,
        x: TensorValue,
        mesh: DeviceMesh,
        dim_specs: List[DimSpec],
        **kwargs: Any,
    ) -> List[TensorValue]:
        """Create sharded TensorValues by slicing the input."""
        from ...core.sharding.spec import ShardingSpec

        # Determine global shape
        global_shape = kwargs.pop('global_shape', None)
        if global_shape is None:
            # Fallback: compute from x.type.shape (only correct for unsharded inputs)
            global_shape = tuple(int(d) for d in x.type.shape)
        
        spec = ShardingSpec(mesh, dim_specs)
        
        # DISTRIBUTED SPMD MODE (called per-shard internally by Operation.__call__)
        if "shard_idx" in kwargs:
             shard_idx = kwargs["shard_idx"]
             return self._slice_for_device(x, global_shape, spec, shard_idx, mesh)
        
        # SIMULATION MODE (manual loop)
        return self._simulate_shard_execution(x, global_shape, spec, mesh)

    def _simulate_shard_execution(self, x, global_shape, spec, mesh):
        """Execute sharding manually for all devices (simulation)."""
        num_shards = len(mesh.devices)
        shard_values = []
        for shard_idx in range(num_shards):
            val = self._slice_for_device(x, global_shape, spec, shard_idx, mesh)
            
            if mesh.is_distributed:
                val = ops.transfer_to(val, mesh.device_refs[shard_idx])
            shard_values.append(val)
            
        return shard_values

    def _slice_for_device(self, x, global_shape, spec, shard_idx, mesh):
        from ...core.sharding.spec import compute_local_shape
        from ...core.tensor import Tensor
        
        # Determine effective input value for this shard
        effective_x = x
        input_shard_offset = [0] * len(global_shape)
        
        # Handle Tensor input (Simulation Mode)
        if isinstance(x, Tensor):
            # Hydrate values if needed (realized tensor)
            x.hydrate()
            vals = x._values  # Use raw after hydrate for indexing
            
            if vals:
                # Eager mode: extract underlying value(s)
                if len(vals) > shard_idx:
                    # Input is sharded/distributed, pick corresponding shard
                    effective_x = vals[shard_idx]
                    
                    # If input had sharding, we must compute its global offset 
                    if x.sharding:
                        for d, dim_spec in enumerate(x.sharding.dim_specs):
                            # Calculate global offset for this dimension on this shard
                            offset = 0
                            shard_pos = 0
                            total_shards = 1
                            for axis in dim_spec.axes:
                                size = mesh.get_axis_size(axis)
                                coord = mesh.get_coordinate(shard_idx, axis)
                                shard_pos = (shard_pos * size) + coord
                                total_shards *= size
                            
                            # Assuming uniform chunking (standard SPMD)
                            dim_global_len = int(global_shape[d])
                            chunk_size = math.ceil(dim_global_len / total_shards)
                            input_shard_offset[d] = shard_pos * chunk_size
                else:
                    # Input is unsharded/replicated (single value), broadcast it
                    effective_x = vals[0]

        # Target local shape for this device
        target_local_shape = compute_local_shape(global_shape, spec, shard_idx)
        
        slices = []
        for d, (t_len, g_len) in enumerate(zip(target_local_shape, global_shape)):
            # Use effective input (local shard) shape
            inp_len = int(effective_x.type.shape[d])
            
            if inp_len == t_len:
                # Identity slice (input matches target)
                slices.append(slice(0, t_len))
                continue
            
            # Compute slice range in GLOBAL coordinates
            dim_spec = spec.dim_specs[d]
            total_shards = 1
            my_shard_pos = 0
            for axis_name in dim_spec.axes:
                size = mesh.get_axis_size(axis_name)
                coord = mesh.get_coordinate(shard_idx, axis_name)
                my_shard_pos = (my_shard_pos * size) + coord
                total_shards *= size
            
            chunk_size = math.ceil(g_len / total_shards)
            start_global = my_shard_pos * chunk_size
            start_global = min(start_global, g_len)
            
            end_global = min(start_global + chunk_size, g_len)
            
            # Map GLOBAL coordinates to LOCAL input coordinates
            # local_start = global_start - input_shard_offset
            # Clip to valid input range [0, inp_len]
            
            start_local = start_global - input_shard_offset[d]
            end_local = end_global - input_shard_offset[d]
            
            # Ensure indices are valid for local input
            start_local = max(0, min(start_local, inp_len))
            end_local = max(0, min(end_local, inp_len))
            
            slices.append(slice(start_local, end_local))
                 
        return effective_x[tuple(slices)]

    def __call__(self, x, mesh: DeviceMesh, dim_specs: List[DimSpec], replicated_axes: Optional[Set[str]] = None, _bypass_idempotency: bool = False):
        """Shard a tensor according to the given specification.
        
        This operation is IDEMPOTENT: if the tensor is already sharded with the
        target spec, it returns the input unchanged (identity). If the tensor
        has a different sharding, it performs proper resharding.
        
        This enables internal shard() calls inside shard_map functions to work
        correctly without causing double-execution.
        
        Args:
            _bypass_idempotency: Internal flag used by reshard_tensor to avoid
                                 recursion. Do not set directly.
        """
        from ...core import Tensor
        from ...core import GRAPH
        from ...core.sharding.spec import ShardingSpec, needs_reshard
        from max import graph as g
        
        target_spec = ShardingSpec(mesh, dim_specs, replicated_axes=replicated_axes or set())
        
        # IDEMPOTENCY CHECK: If input is already correctly sharded, return identity
        # Skip this check when called from reshard_tensor to avoid recursion
        if not _bypass_idempotency and isinstance(x, Tensor) and x.sharding:
            if not needs_reshard(x.sharding, target_spec):
                # Already correctly sharded - return identity (no-op)
                return x
            
            # Different sharding - need to reshard via all_gather + shard
            # This handles the case where input is sharded on 'dp' but we want 'tp'
            from ...core.sharding.spmd import reshard_tensor
            return reshard_tensor(x, x.sharding, target_spec, mesh)
        
        # Standard path: input is unsharded, shard it according to spec
        
        # Compute global shape BEFORE sharding (from input tensor's local + sharding)
        global_shape = None
        if isinstance(x, Tensor):
            # For sharded inputs, compute global from local + sharding
            local = x.physical_local_shape(0)
            if local is not None and x.sharding:
                from ...core.sharding.spec import compute_global_shape
                global_shape = compute_global_shape(tuple(local), x.sharding)
            elif local is not None:
                global_shape = tuple(int(d) for d in local)
        
        # Hydrate values from storages if needed (MUST be before graph context)
        if isinstance(x, Tensor):
            x.hydrate()
        
        with GRAPH.graph:
            # Convert input to TensorValue (lazy) or keep as Tensor (eager simulation)
            x_input = x
            if isinstance(x, Tensor) and not x._values:
                 # No values even after hydrate - use TensorValue
                 x_input = g.TensorValue(x)
            
            # Execute shard operation with global_shape
            # maxpr and _slice_for_device will handle picking the correct shard and slicing it
            shard_values = self.maxpr(x_input, mesh, dim_specs, global_shape=global_shape)
        
        # Create output tensor with multiple values
        spec = ShardingSpec(mesh, dim_specs, replicated_axes=replicated_axes or set())
        output = Tensor._create_unsafe(
            values=shard_values,
            traced=x.traced if isinstance(x, Tensor) else False,
            batch_dims=x.batch_dims if isinstance(x, Tensor) else 0,
        )
        output.sharding = spec
        
        # NABLA 2026: Cached metadata removed.
        
        # Setup tracing refs for graph traversal
        traced = x.traced if isinstance(x, Tensor) else False
        self._setup_output_refs(output, (x,), {'mesh': mesh, 'dim_specs': dim_specs}, traced)
        
        return output
    
    
    def _compute_global_from_local(self, local_shape, sharding):
        """Deprecated: use spmd.compute_global_shape."""
        from ...core.sharding.spec import compute_global_shape
        return compute_global_shape(local_shape, sharding)


# Singleton instance
shard_op = ShardOp()

# Public API
def shard(x, mesh: DeviceMesh, dim_specs: List[DimSpec], **kwargs):
    """Shard a tensor according to the given mesh and dimension specs.
    
    Args:
        x: Input tensor (replicated/unsharded)
        mesh: Device mesh defining shard topology
        dim_specs: List of DimSpec for each dimension
        **kwargs: Internal arguments (e.g. _bypass_idempotency)
        
    Returns:
        Sharded tensor with multiple internal TensorValues
    """
    return shard_op(x, mesh, dim_specs, **kwargs)
