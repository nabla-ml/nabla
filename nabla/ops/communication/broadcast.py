# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

from max.graph import TensorValue, ops

from ..base import OpArgs, OpKwargs, OpResult, OpTensorValues
from .base import CollectiveOperation


class DistributedBroadcastOp(CollectiveOperation):
    """Broadcast a tensor from a root device to all other devices in the mesh."""

    @property
    def name(self) -> str:
        return "distributed_broadcast"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Output shape is same as input shape on all shards (replicated)."""
        from ...core.sharding import spmd

        x = args[0]
        mesh = self._derive_mesh(x, kwargs) or spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else x.num_shards

        # In a distributed broadcast, all outputs have the same shape as the input at root.
        # We assume input rank is matching.
        s = x.physical_local_shape(0)
        if s is None:
            s = x.shape

        shapes = [tuple(int(d) for d in s)] * num_shards
        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def execute(
        self, args: OpArgs, kwargs: OpKwargs
    ) -> tuple[list[TensorValue], ShardingSpec | None, DeviceMesh | None]:
        """Execute distributed broadcast using MAX ops."""
        from ...core import GRAPH, Tensor
        from ...core.sharding.spmd import create_replicated_spec

        x: Tensor = args[0]
        mesh = self._derive_mesh(x, kwargs)

        with GRAPH.graph:
            if mesh and mesh.is_distributed:
                root = x.values[0]
                root_device = mesh.device_refs[0]
                if root.type.device != root_device:
                    root = ops.transfer_to(root, root_device)

                try:
                    from max.graph.ops import (
                        distributed_broadcast as max_distributed_broadcast,
                    )
                    from max.dtype import DType
                    from max.graph.type import BufferType
                except ImportError as exc:
                    raise RuntimeError(
                        "Native MAX op 'distributed_broadcast' is required for "
                        "distributed mesh broadcast but is not available in this "
                        "MAX build. Install nightly modular/max versions."
                    ) from exc

                signal_buffers = [
                    ops.buffer_create(BufferType(DType.uint8, (65536,), dev))
                    for dev in mesh.device_refs
                ]
                results = max_distributed_broadcast(root, signal_buffers)
            else:
                # Simulation: just replicate the first shard
                results = [x.values[0]] * (len(mesh.devices) if mesh else 1)

        rank = len(x.shape)
        output_spec = create_replicated_spec(mesh, rank)

        return (results, output_spec, mesh)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for distributed broadcast: AllReduce(sum) back to root."""
        from .all_reduce import all_reduce

        return [all_reduce(cotangents[0])]


_distributed_broadcast_op = DistributedBroadcastOp()


def distributed_broadcast(x, mesh=None):
    """Broadcast a tensor across a distributed mesh."""
    return _distributed_broadcast_op([x], {"mesh": mesh})[0]
