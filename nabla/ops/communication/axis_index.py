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


class AxisIndexOp(CollectiveOperation):
    """Return the device's position along a mesh axis."""

    @property
    def name(self) -> str:
        return "axis_index"

    def physical_execute(self, args: tuple[Any, ...], kwargs: dict) -> Any:
        """Return the device's position along a mesh axis (Physical)."""
        from ...core import GRAPH, Tensor
        from ...core.sharding.spec import DimSpec, ShardingSpec
        from max.dtype import DType

        mesh = self._derive_mesh(None, kwargs)
        axis_name = kwargs.get("axis_name")

        if mesh is None or axis_name is None:
            # AxisIndex might be called with positional args
            if len(args) >= 2:
                mesh, axis_name = args[0], args[1]
            else:
                raise ValueError("AxisIndexOp requires 'mesh' and 'axis_name' arguments.")

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


axis_index_op = AxisIndexOp()


def axis_index(mesh: DeviceMesh, axis_name: str):
    """Return each device's position along a mesh axis."""
    return axis_index_op(mesh, axis_name)
