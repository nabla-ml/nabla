# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

from max.graph import TensorValue, ops

from .base import CollectiveOperation


class DistributedBroadcastOp(CollectiveOperation):
    """Broadcast a tensor from a root device to all other devices in the mesh."""

    @property
    def name(self) -> str:
        return "distributed_broadcast"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
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
        dtypes = [x.dtype] * num_shards
        
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

        return shapes, dtypes, devices

    def execute(self, args: tuple[Any, ...], kwargs: dict) -> Any:
        """Execute distributed broadcast using MAX ops."""
        from ...core import GRAPH, Tensor
        from ...core.sharding.spmd import create_replicated_spec

        x: Tensor = args[0]
        mesh = self._derive_mesh(x, kwargs)

        with GRAPH.graph:
            if mesh and mesh.is_distributed:
                signal_buffers = mesh.get_signal_buffers()
                # Input assumed to be on root device (first device in mesh for now)
                # If it's already sharded/replicated, MAX handles it.
                results = ops.distributed_broadcast(x.values[0], signal_buffers)
            else:
                # Simulation: just replicate the first shard
                results = [x.values[0]] * (len(mesh.devices) if mesh else 1)

        rank = len(x.shape)
        output_spec = create_replicated_spec(mesh, rank)
        
        return (results, output_spec, mesh)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for distributed broadcast: AllReduce(sum) back to root."""
        from .all_reduce import all_reduce
        return (all_reduce(cotangent),)


distributed_broadcast_op = DistributedBroadcastOp()


def distributed_broadcast(x, mesh=None):
    """Broadcast a tensor across a distributed mesh."""
    return distributed_broadcast_op(x, mesh=mesh)
