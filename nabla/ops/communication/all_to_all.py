# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh


class AllToAllOp(CollectiveOperation):
    """All-to-all collective (distributed transpose).

    Each device splits its tensor along split_axis, sends parts to other devices,
    receives from all, and concatenates along concat_axis.
    """

    @property
    def name(self) -> str:
        return "all_to_all"

    def maxpr(
        self,
        shard_values: list[TensorValue],
        split_axis: int,
        concat_axis: int,
        mesh: DeviceMesh = None,
        tiled: bool = True,
    ) -> list[TensorValue]:
        """All-to-all: distributed transpose of tensor blocks."""
        num_devices = len(shard_values)

        if num_devices <= 1:
            return shard_values

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

        received_per_device = []
        for dst in range(num_devices):
            received = []
            for src in range(num_devices):
                chunk = chunks_per_device[src][dst]

                if mesh and mesh.is_distributed:
                    chunk = ops.transfer_to(chunk, mesh.device_refs[dst])

                received.append(chunk)
            received_per_device.append(received)

        results = []
        for dst in range(num_devices):
            if tiled:
                concatenated = ops.concat(received_per_device[dst], axis=concat_axis)
            else:
                concatenated = ops.stack(received_per_device[dst], axis=concat_axis)
            results.append(concatenated)

        return results

    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output sharding: Swap sharding from split_axis to concat_axis is implied?
        Actually:
        Input sharded on concat_axis (usually).
        split_axis is split => becomes sharded.
        concat_axis is concated => becomes replicated.

        So we swap the specs of split_axis and concat_axis?
        Or rather:
        spec[split_axis] += sharded_mesh_axis
        spec[concat_axis] -= sharded_mesh_axis
        """
        from ...core.sharding.spec import DimSpec, ShardingSpec

        mesh = input_tensor.sharding.mesh if input_tensor.sharding else None
        input_spec = input_tensor.sharding

        if mesh and input_spec:
            split_axis = kwargs.get("split_axis", 0)
            concat_axis = kwargs.get("concat_axis", 0)

            new_dim_specs = [
                DimSpec(list(ds.axes), is_open=ds.is_open)
                for ds in input_spec.dim_specs
            ]

            source_axes = new_dim_specs[concat_axis].axes
            target_axes = new_dim_specs[split_axis].axes

            if source_axes:

                moved_axes = list(source_axes)
                new_dim_specs[concat_axis] = DimSpec([], is_open=True)
                new_dim_specs[split_axis] = DimSpec(
                    sorted(list(set(target_axes) | set(moved_axes)))
                )

            return ShardingSpec(mesh, new_dim_specs)

        return None


all_to_all_op = AllToAllOp()


def all_to_all(sharded_tensor, split_axis: int, concat_axis: int, tiled: bool = True):
    """All-to-all collective (distributed transpose)."""
    return all_to_all_op(
        sharded_tensor, split_axis=split_axis, concat_axis=concat_axis, tiled=tiled
    )
