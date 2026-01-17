# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Set

from max.graph import TensorValue, ops

from ..base import Operation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec


class CollectiveOperation(Operation):
    """Base class for collective communication operations.
    
    Reduces boilerplate for operations that:
    1. Take a sharded/replicated tensor as input
    2. Hydrate its values
    3. Perform a MAX graph operation (maxpr)
    4. Return a new Tensor with updated sharding spec
    
    Also provides unified cost modeling infrastructure.
    """
    
    def __call__(self, sharded_tensor, **kwargs):
        from ...core import Tensor
        from ...core import TensorImpl
        from ...core import GRAPH
        
        # 1. Validation and early exit
        if not self._should_proceed(sharded_tensor):
            return sharded_tensor
            
        mesh = sharded_tensor._impl.sharding.mesh if sharded_tensor._impl.sharding else None
        
        # Hydrate values from storages if needed
        sharded_tensor.hydrate()
        
        # 2. Execution in graph context
        with GRAPH.graph:
            # Filter kwargs for maxpr
            maxpr_kwargs = {k: v for k, v in kwargs.items() if k not in ('mesh', 'reduce_axes')}
            result_values = self.maxpr(sharded_tensor.values, mesh=mesh, **maxpr_kwargs)
            
        # 3. Output wrapping
        output_spec = self._compute_output_spec(sharded_tensor, result_values, **kwargs)
        
        impl = TensorImpl(
            values=result_values,
            traced=sharded_tensor._impl.traced,
            batch_dims=sharded_tensor._impl.batch_dims,
        )
        impl.sharding = output_spec
        # NABLA 2026: Cached metadata removed. Global shape computed on demand.
        
        output = Tensor(impl=impl)
        
        # 4. Tracing setup
        self._setup_output_refs(output, (sharded_tensor,), kwargs, sharded_tensor._impl.traced)
        
        return output

    def _should_proceed(self, tensor):
        """Check if operation should proceed (has sharding and potentially multiple shards)."""
        if not tensor._impl.sharding:
            return False
        # If has storages/values, check if > 1 (distributed/sharded) or if we need to enforce algo
        if (tensor._impl._values and len(tensor._impl._values) > 1) or \
           (tensor._impl._storages and len(tensor._impl._storages) > 1):
            return True
        return False
        
    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Compute output sharding spec. Default: preserve input spec."""
        return input_tensor._impl.sharding

    def communication_cost(
        self, 
        input_specs: list["ShardingSpec"], 
        output_specs: list["ShardingSpec"], 
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        mesh: "DeviceMesh"
    ) -> float:
        """Unified communication cost estimation.
        
        Delegates to self.estimate_cost() which must be implemented by subclasses.
        Calculates basic tensor size metrics to pass to estimate_cost.
        """
        if not input_shapes:
            return 0.0
            
        # Calculate tensor size in bytes
        # Assumes float32 (4 bytes) by default
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        size_bytes = num_elements * 4
        
        # Extract axes info if possible
        # This is heuristics-based as specific op logic differs
        axes = []
        if input_specs and input_specs[0]:
            # Collect all sharded axes from input
            for dim_spec in input_specs[0].dim_specs:
                axes.extend(dim_spec.axes)
        
        return self.estimate_cost(size_bytes, mesh, axes, input_specs, output_specs)

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: "DeviceMesh",
        axes: list[str],
        input_specs: list["ShardingSpec"] = None,
        output_specs: list["ShardingSpec"] = None,
    ) -> float:
        """Estimate cost of the collective operation.
        
        Args:
            size_bytes: Total size of the tensor in bytes
            mesh: Device mesh
            axes: Relevant mesh axes for this operation
            input_specs: Input sharding specs (optional context)
            output_specs: Output sharding specs (optional context)
        """
        return 0.0

    def _group_shards_by_axes(self, shard_values, mesh, group_by_axes):
        """Group shards by coordinates on specific axes.
        
        Args:
            shard_values: List of shard values (one per device)
            mesh: Device mesh
            group_by_axes: List/set of axis names to use for grouping. 
                           Devices with same coords on these axes will be in same group.
        
        Returns:
            Dict mapping coord_tuple -> list of (shard_idx, value)
        """
        groups = {}
        for shard_idx, val in enumerate(shard_values):
            # Build key from coords on grouping axes
            key_parts = []
            for axis_name in group_by_axes:
                key_parts.append(mesh.get_coordinate(shard_idx, axis_name))
            
            key = tuple(key_parts)
            if key not in groups:
                 groups[key] = []
            groups[key].append((shard_idx, val))
            
        return groups
