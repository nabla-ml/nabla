# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import Operation
from .all_gather import AllGatherOp, all_gather
from .shard import shard_op

if TYPE_CHECKING:
    from ...core import Tensor
    from ...core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec


class ReshardOp(Operation):
    """Generic resharding operation.

    Reshards a tensor from its current sharding (or replication) to a new target
    sharding specification. Handles both logical to physical spec conversion
    and data movement.
    """

    @property
    def name(self) -> str:
        return "reshard"

    def communication_cost(
        self,
        input_specs: list[ShardingSpec],
        output_specs: list[ShardingSpec],
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        mesh: DeviceMesh,
    ) -> float:
        """Estimate cost of resharding."""
        from_spec = input_specs[0] if input_specs else None
        to_spec = output_specs[0] if output_specs else None

        if not input_shapes:
            return 0.0

        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        tensor_bytes = num_elements * 4

        if from_spec is None and to_spec is None:
            return 0.0

        if from_spec is None:

            return 0.0

        if to_spec is None:

            axes_to_gather = set()
            for dim_spec in from_spec.dim_specs:
                axes_to_gather.update(dim_spec.axes)

            total_shards = from_spec.total_shards
            local_bytes = tensor_bytes // (total_shards or 1)

            return AllGatherOp.estimate_cost(local_bytes, mesh, list(axes_to_gather))

        total_cost = 0.0

        if len(from_spec.dim_specs) != len(to_spec.dim_specs):
            return float("inf")

        for from_dim, to_dim in zip(
            from_spec.dim_specs, to_spec.dim_specs, strict=False
        ):
            from_axes = set(from_dim.axes)
            to_axes = set(to_dim.axes)

            removed_axes = from_axes - to_axes
            if removed_axes:
                from_shards = 1
                for axis in from_dim.axes:
                    from_shards *= mesh.get_axis_size(axis)

                local_bytes_dim = tensor_bytes // from_shards

                total_cost += AllGatherOp.estimate_cost(
                    local_bytes_dim, mesh, list(removed_axes)
                )

        return total_cost

    def maxpr(self, *args, **kwargs):
        """ReshardOp is a composite operation. Use __call__ instead."""
        raise NotImplementedError(
            "ReshardOp is a composite operation. "
            "Use __call__ which orchestrates all_gather + shard_op."
        )

    def execute(
        self,
        tensor: Tensor,
        mesh: DeviceMesh,
        dim_specs: list[DimSpec],
        replicated_axes: set[str] | None = None,
    ) -> Tensor:
        """Reshard tensor to target specs."""
        from ...core.sharding.spec import DimSpec, ShardingSpec, needs_reshard

        batch_dims = tensor.batch_dims
        current_rank = len(tensor.shape)

        if batch_dims > 0:
            if len(dim_specs) == current_rank:
                batch_specs = [DimSpec([], is_open=True) for _ in range(batch_dims)]

                if tensor.sharding:
                    current_s = tensor.sharding
                    if len(current_s.dim_specs) >= batch_dims:
                        for i in range(batch_dims):
                            batch_specs[i] = current_s.dim_specs[i].clone()

                dim_specs = batch_specs + list(dim_specs)
            elif len(dim_specs) != (current_rank + batch_dims):
                pass

        target_spec = ShardingSpec(
            mesh, dim_specs, replicated_axes=replicated_axes or set()
        )
        current_spec = tensor.sharding

        if not needs_reshard(current_spec, target_spec):
            if current_spec is None:
                tensor.sharding = target_spec
            return tensor

        result = tensor
        if current_spec:
            for dim in range(len(current_spec.dim_specs)):
                from_axes = (
                    set(current_spec.dim_specs[dim].axes)
                    if dim < len(current_spec.dim_specs)
                    else set()
                )
                to_axes = (
                    set(target_spec.dim_specs[dim].axes)
                    if dim < len(target_spec.dim_specs)
                    else set()
                )

                axes_to_remove = from_axes - to_axes
                if axes_to_remove:
                    result = all_gather(result, axis=dim)

        result = shard_op(
            result,
            mesh,
            target_spec.dim_specs,
            replicated_axes=target_spec.replicated_axes,
        )

        return result


reshard_op = ReshardOp()


def reshard(
    tensor: Tensor,
    mesh: DeviceMesh,
    dim_specs: list[DimSpec],
    replicated_axes: set[str] | None = None,
    **kwargs,
) -> Tensor:
    """Reshard tensor to target specs."""
    return reshard_op(tensor, mesh, dim_specs, replicated_axes, **kwargs)
