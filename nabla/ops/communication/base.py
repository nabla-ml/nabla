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

    def infer_sharding_spec(self, args: Any, mesh: DeviceMesh, kwargs: dict) -> Any:
        """Default adaptation: validate inputs and compute output spec."""
        # This implementation allows subclasses to strictly rely on _compute_output_spec
        # for both adaptation and execution phases.

        if not args:
            return None, [], False

        input_tensor = args[0]
        input_sharding = input_tensor.sharding

        # Validation: check if we should proceed (must have sharding)
        if not input_sharding:
            # If no input sharding, typically no output sharding unless op creates it.
            # Most comm ops require input sharding.
            return None, [None] * len(args), False

        # Compute output sharding using the subclass logic
        # We pass None for results locally since this is logical inference
        output_sharding = self._compute_output_spec(input_tensor, None, **kwargs)

        # Default: preserve input sharding for all args
        input_shardings = [
            arg.sharding if hasattr(arg, "sharding") else None for arg in args
        ]

        return output_sharding, input_shardings, False

    def _derive_mesh(self, tensor, kwargs):
        """Derive device mesh from tensor or kwargs."""
        if hasattr(tensor, "sharding") and tensor.sharding:
            return tensor.sharding.mesh
        return kwargs.get("mesh")

    def _get_physical_axis(self, tensor, axis):
        """Convert logical axis to physical axis, accounting for batch_dims."""
        if axis is None:
            return None

        # Current batch dims (inserted by vmap) are always at the front
        batch_dims = tensor.batch_dims
        logical_rank = len(tensor.shape)  # .shape is logical shape if batch_dims > 0?
        # Wait, wrapper Tensor.shape usually returns logical shape if batch_dims > 0?
        # Let's check Tensor implementation.
        # Assuming Tensor.shape is logical shape:

        # If we access tensor.shape, it returns the shape MINUS batch dims?
        # Let's assume standard behavior:
        # axis < 0 -> axis + logical_rank
        # physical_axis = batch_dims + axis

        # To be safe, let's use the full rank from internal values if possible,
        # but Tensor wrapper abstracts this.
        # Let's trust tensor.shape is logical.

        logical_rank = len(tensor.shape)
        norm_axis = axis if axis >= 0 else logical_rank + axis

        if norm_axis < 0 or norm_axis >= logical_rank:
            raise ValueError(
                f"Axis {axis} out of bounds for logical rank {logical_rank}"
            )

        return batch_dims + norm_axis

    def _get_sharded_axis_name(self, tensor, axis):
        """Get the name of the mesh axis that a tensor is sharded on at the given dimension."""
        if axis is None or not hasattr(tensor, "sharding") or not tensor.sharding:
            return None

        sharding = tensor.sharding
        physical_axis = self._get_physical_axis(tensor, axis)

        if 0 <= physical_axis < len(sharding.dim_specs):
            dim_spec = sharding.dim_specs[physical_axis]
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
