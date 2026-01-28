# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..base import Operation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, ShardingSpec


class CollectiveOperation(Operation):
    """Base class for collective communication operations.

    Handles value hydration, graph execution (maxpr), and output wrapping/sharding update.
    """

    # Legacy execute and maxpr_all methods have been removed. 
    # All communication operations now implement physical_execute.

    def _derive_mesh(self, tensor, kwargs):
        """Derive device mesh from tensor or kwargs."""
        if hasattr(tensor, "sharding") and tensor.sharding:
            return tensor.sharding.mesh
        return kwargs.get("mesh")

    def _get_sharded_axis_name(self, tensor, axis):
        """Get the name of the mesh axis that a tensor is sharded on at the given dimension."""
        if axis is None or not hasattr(tensor, "sharding") or not tensor.sharding:
            return None
        
        sharding = tensor.sharding
        # Normalize axis
        rank = len(sharding.dim_specs)
        spec_idx = axis if axis >= 0 else rank + axis
        
        if 0 <= spec_idx < rank:
            dim_spec = sharding.dim_specs[spec_idx]
            if dim_spec.axes:
                return dim_spec.axes[0]  # Standard 1D-per-dim sharding
        return None

    def _get_reduce_axes(self, tensor, kwargs):
        """Determine which mesh axes to reduce over."""
        reduce_axes = kwargs.get("reduce_axes")
        if reduce_axes is not None:
             if isinstance(reduce_axes, str):
                 return {reduce_axes}
             return set(reduce_axes)
        
        # Fallback: reduce over all partial axes if none specified
        if hasattr(tensor, "sharding") and tensor.sharding:
            if tensor.sharding.partial_sum_axes:
                return set(tensor.sharding.partial_sum_axes)
        return None

    def _should_proceed(self, tensor):
        """Check if operation should proceed (has sharding and potentially multiple shards)."""
        if not tensor.sharding:
            return False
        if (tensor._values and len(tensor._values) > 1) or (
            tensor._storages and len(tensor._storages) > 1
        ):
            return True
        return False

    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Compute output sharding spec. Default: preserve input spec."""
        return input_tensor.sharding

    def communication_cost(
        self,
        input_specs: list[ShardingSpec],
        output_specs: list[ShardingSpec],
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        mesh: DeviceMesh,
    ) -> float:
        """Unified communication cost estimation."""
        if not input_shapes:
            return 0.0

        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        size_bytes = num_elements * 4

        axes = []
        if input_specs and input_specs[0]:
            for dim_spec in input_specs[0].dim_specs:
                axes.extend(dim_spec.axes)

        return self.estimate_cost(size_bytes, mesh, axes, input_specs, output_specs)

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: DeviceMesh,
        axes: list[str],
        input_specs: list[ShardingSpec] = None,
        output_specs: list[ShardingSpec] = None,
    ) -> float:
        """Estimate cost of the collective operation."""
        return 0.0

    def _group_shards_by_axes(self, shard_values, mesh, group_by_axes):
        """Group shards by coordinates on specific axes."""
        groups = {}
        for shard_idx, val in enumerate(shard_values):
            key_parts = []
            for axis_name in group_by_axes:
                key_parts.append(mesh.get_coordinate(shard_idx, axis_name))

            key = tuple(key_parts)
            if key not in groups:
                groups[key] = []
            groups[key].append((shard_idx, val))

        return groups
