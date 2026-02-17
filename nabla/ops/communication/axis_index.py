# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, ShardingSpec
    from ..base import OpArgs, OpKwargs


class AxisIndexOp(CollectiveOperation):
    """Return the device's position along a mesh axis."""

    @property
    def name(self) -> str:
        return "axis_index"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for axis_index (scalar per shard)."""
        from max.dtype import DType

        mesh = self._derive_mesh(None, kwargs)
        if mesh is None and len(args) >= 1:
            mesh = args[0]

        num_shards = len(mesh.devices) if mesh else 1
        shapes = [(1,)] * num_shards

        dtypes = [DType.int32] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = list(mesh.device_refs)
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [None] * num_shards  # Scalars/unplaced

        return shapes, dtypes, devices

    def execute(
        self, args: OpArgs, kwargs: OpKwargs
    ) -> tuple[list[TensorValue], ShardingSpec | None, DeviceMesh | None]:
        """Return the device's position along a mesh axis (Physical)."""
        from max.dtype import DType

        from ...core import GRAPH
        from ...core.sharding.spec import DimSpec, ShardingSpec

        mesh = self._derive_mesh(None, kwargs)
        axis_name = kwargs.get("axis_name")

        if mesh is None or axis_name is None:
            # AxisIndex might be called with positional args
            if len(args) >= 2:
                mesh, axis_name = args[0], args[1]
            else:
                raise ValueError(
                    "AxisIndexOp requires 'mesh' and 'axis_name' arguments."
                )

        results = []
        with GRAPH.graph:
            for shard_idx in range(len(mesh.devices)):
                coord = mesh.get_coordinate(shard_idx, axis_name)
                device = mesh.device_refs[shard_idx] if mesh.device_refs else None
                val = ops.constant(coord, DType.int32, device)
                val = ops.reshape(val, (1,))
                results.append(val)

        output_spec = ShardingSpec(mesh, [DimSpec([axis_name])])
        return (results, output_spec, mesh)


_axis_index_op = AxisIndexOp()


def axis_index(mesh: DeviceMesh, axis_name: str):
    """Return each device's position along a mesh axis."""
    return _axis_index_op([], {"mesh": mesh, "axis_name": axis_name})[0]
