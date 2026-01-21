# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from ..base import Operation
from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, ShardingSpec


class AllGatherOp(CollectiveOperation):
    """Gather shards along an axis to produce replicated full tensors."""

    @property
    def name(self) -> str:
        return "all_gather"

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: DeviceMesh,
        axes: list[str],
        input_specs: list[ShardingSpec] = None,
        output_specs: list[ShardingSpec] = None,
    ) -> float:
        """Estimate AllGather cost."""
        if not axes:
            return 0.0

        n_devices = 1
        for axis in axes:
            n_devices *= mesh.get_axis_size(axis)

        if n_devices <= 1:
            return 0.0

        local_bytes = size_bytes // n_devices

        bandwidth = getattr(mesh, "bandwidth", 1.0)
        cost = (n_devices - 1) / n_devices * size_bytes / bandwidth
        return cost

    def maxpr(
        self,
        shard_values: list[TensorValue],
        axis: int,
        mesh: DeviceMesh = None,
        sharded_axis_name: str = None,
    ) -> list[TensorValue]:
        """Gather all shards along the specified axis."""

        if mesh and mesh.is_distributed:
            from max.dtype import DType
            from max.graph.ops.allgather import allgather as max_allgather
            from max.graph.type import BufferType

            # Signal buffer must be uint8 and large enough (>49KB) to avoid errors
            BUFFER_SIZE = 65536
            signal_buffers = [
                ops.buffer_create(BufferType(DType.uint8, (BUFFER_SIZE,), dev))
                for dev in mesh.device_refs
            ]
            return max_allgather(shard_values, signal_buffers, axis=axis)

        if mesh is None or sharded_axis_name is None:

            if len(shard_values) == 1:
                return shard_values
            full_tensor = ops.concat(shard_values, axis=axis)
            return [full_tensor] * len(shard_values)

        return self._simulate_grouped_gather(
            shard_values, axis, mesh, sharded_axis_name
        )

    def _simulate_grouped_gather(self, shard_values, axis, mesh, sharded_axis_name):
        """Handle simulated gather logic for multi-dimensional meshes."""

        all_axes = list(mesh.axis_names)
        other_axes = [ax for ax in all_axes if ax != sharded_axis_name]

        if not other_axes:

            unique_shards = []
            seen_coords = set()
            for shard_idx, val in enumerate(shard_values):
                coord = mesh.get_coordinate(shard_idx, sharded_axis_name)
                if coord not in seen_coords:
                    seen_coords.add(coord)
                    unique_shards.append((coord, val))
            unique_shards.sort(key=lambda x: x[0])
            shards_to_concat = [v for _, v in unique_shards]

            full_tensor = (
                ops.concat(shards_to_concat, axis=axis)
                if len(shards_to_concat) > 1
                else shards_to_concat[0]
            )
            return [full_tensor] * len(shard_values)

        groups = self._group_shards_by_axes(shard_values, mesh, other_axes)

        gathered_per_group = {}
        for other_coords, members in groups.items():

            members.sort(key=lambda x: mesh.get_coordinate(x[0], sharded_axis_name))
            shards = [val for _, val in members]
            gathered = ops.concat(shards, axis=axis) if len(shards) > 1 else shards[0]
            gathered_per_group[other_coords] = gathered

        results = []
        for shard_idx in range(len(shard_values)):
            other_coords = tuple(
                mesh.get_coordinate(shard_idx, ax) for ax in other_axes
            )
            results.append(gathered_per_group[other_coords])
        return results

    def __call__(self, sharded_tensor, axis: int, **kwargs):
        """Gather all shards to produce a replicated tensor."""
        from ...core import GRAPH, Tensor
        from ...core.sharding.spec import DimSpec, ShardingSpec

        if (not sharded_tensor._values and not sharded_tensor._storages) or (
            sharded_tensor.sharding and sharded_tensor.sharding.is_fully_replicated()
        ):
            return sharded_tensor

        mesh = sharded_tensor.sharding.mesh if sharded_tensor.sharding else None
        sharded_axis_name = None

        if sharded_tensor.sharding:

            sharding = sharded_tensor.sharding
            if axis < len(sharding.dim_specs) and sharding.dim_specs[axis].axes:
                sharded_axis_name = sharding.dim_specs[axis].axes[0]

        with GRAPH.graph:
            sharded_tensor.hydrate()

            if len(sharded_tensor.values) <= 1:
                from max.graph import Shape

                from ...core.sharding.spec import compute_global_shape

                batch_dims = sharded_tensor.batch_dims

                local_shape = sharded_tensor.physical_local_shape(0)
                if local_shape is None:
                    local_shape = sharded_tensor.shape

                if sharded_tensor.sharding and local_shape is not None:
                    global_shape_tuple = compute_global_shape(
                        tuple(local_shape), sharded_tensor.sharding
                    )
                    global_shape = Shape(global_shape_tuple)
                else:
                    global_shape = local_shape

                rank = len(global_shape) if global_shape else len(sharded_tensor.shape)
                replicated_spec = (
                    ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])
                    if mesh
                    else None
                )

                tensor = Tensor._create_unsafe(
                    storages=sharded_tensor._storages,
                    values=sharded_tensor._values,
                    traced=sharded_tensor.traced,
                    batch_dims=batch_dims,
                )
                tensor.sharding = replicated_spec

                return tensor

        with GRAPH.graph:
            gathered = self.maxpr(
                sharded_tensor.values,
                axis,
                mesh=mesh,
                sharded_axis_name=sharded_axis_name,
            )

        from max.graph import Shape

        from ...core.sharding.spec import compute_global_shape

        local_shape = sharded_tensor.physical_local_shape(0)
        if sharded_tensor.sharding and local_shape is not None:
            global_shape_tuple = compute_global_shape(
                tuple(local_shape), sharded_tensor.sharding
            )
            global_shape = Shape(global_shape_tuple)
        else:

            global_shape = None

        output_spec = None
        if mesh and sharded_tensor.sharding:
            input_spec = sharded_tensor.sharding
            new_dim_specs = []
            for dim_idx, dim_spec in enumerate(input_spec.dim_specs):
                if dim_idx == axis:

                    new_dim_specs.append(DimSpec([]))
                else:

                    new_dim_specs.append(dim_spec)
            output_spec = ShardingSpec(mesh, new_dim_specs)

        output = Tensor._create_unsafe(
            values=gathered,
            traced=sharded_tensor.traced,
            batch_dims=sharded_tensor.batch_dims,
        )
        output.sharding = output_spec

        self._setup_output_refs(
            output, (sharded_tensor,), {"axis": axis}, sharded_tensor.traced
        )

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
        shard_values: list[TensorValue],
        source_spec: ShardingSpec,
        mesh: DeviceMesh = None,
    ) -> TensorValue:
        """Reconstruct the global tensor from potentially multi-dimensional shards.

        Algorithm (hierarchical concatenation):
        1. For each sharded tensor dimension: Group shards by coordinates on ALL mesh axes EXCEPT the one being merged.
        2. Within each group, sort by coordinate on the merge axis and concatenate.
        """
        if source_spec.is_fully_replicated():
            return shard_values[0]

        current_shard_descs = [
            (shard_values[i], source_spec.mesh.devices[i])
            for i in range(len(shard_values))
        ]
        rank = len(shard_values[0].type.shape)

        current_active_axes = set()
        for dim in source_spec.dim_specs:
            current_active_axes.update(dim.axes)
        current_active_axes.update(source_spec.replicated_axes)

        for d in range(rank - 1, -1, -1):
            if d >= len(source_spec.dim_specs):
                continue
            dim_spec = source_spec.dim_specs[d]

            for ax in reversed(dim_spec.axes):

                groups = {}
                for val, device_id in current_shard_descs:
                    signature = []

                    for check_ax in sorted(list(current_active_axes)):
                        if check_ax == ax:
                            continue

                        try:
                            c = source_spec.mesh.get_coordinate(device_id, check_ax)
                            signature.append((check_ax, c))
                        except Exception:

                            continue

                    key = tuple(signature)

                    if key not in groups:
                        groups[key] = []

                    my_coord = source_spec.mesh.get_coordinate(device_id, ax)
                    groups[key].append((my_coord, val, device_id))

                new_shard_descs = []

                for key, members in groups.items():

                    members.sort(key=lambda x: x[0])

                    unique_chunks = []
                    seen_coords = set()

                    for m in members:
                        coord = m[0]
                        if coord not in seen_coords:
                            unique_chunks.append(m[1])
                            seen_coords.add(coord)

                    # Transfer all chunks to the first chunk's device before concatenating
                    if mesh and mesh.is_distributed and len(unique_chunks) > 1:
                        target_device = unique_chunks[0].device
                        unique_chunks = [
                            ops.transfer_to(chunk, target_device) 
                            for chunk in unique_chunks
                        ]
                    
                    merged = ops.concat(unique_chunks, axis=d)

                    new_shard_descs.append((merged, members[0][2]))

                current_shard_descs = new_shard_descs
                current_active_axes.remove(ax)

        return current_shard_descs[0][0]

    def __call__(self, sharded_tensor):
        """Gather all sharded axes to produce a replicated tensor."""

        from ...core import GRAPH, Tensor
        from ...core.sharding.spec import DimSpec, ShardingSpec

        if not sharded_tensor.sharding:
            return sharded_tensor

        spec = sharded_tensor.sharding
        mesh = spec.mesh

        if spec.is_fully_replicated():
            return sharded_tensor

        sharded_tensor.hydrate()
        shard_values = sharded_tensor.values

        if not shard_values or len(shard_values) <= 1:
            return sharded_tensor

        with GRAPH.graph:
            global_tensor = self.maxpr(shard_values, spec, mesh)

        rank = len(global_tensor.type.shape)
        replicated_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])

        tensor = Tensor._create_unsafe(
            values=[global_tensor],
            traced=sharded_tensor.traced,
            batch_dims=sharded_tensor.batch_dims,
        )
        tensor.sharding = replicated_spec
        return tensor


all_gather_op = AllGatherOp()
gather_all_axes_op = GatherAllAxesOp()


def all_gather(sharded_tensor, axis: int, **kwargs):
    """Gather all shards to produce a replicated tensor."""
    return all_gather_op(sharded_tensor, axis, **kwargs)


def gather_all_axes(sharded_tensor):
    """Gather all sharded axes to produce a fully replicated tensor."""
    return gather_all_axes_op(sharded_tensor)
