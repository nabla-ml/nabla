# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from ..base import Operation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, DimSpec


class ShardOp(Operation):
    """Split a replicated tensor into multiple sharded TensorValues."""

    @property
    def name(self) -> str:
        return "shard"

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for shard: reshard back to input's sharding."""
        x = primals[0] if isinstance(primals, (list, tuple)) else primals
        from .reshard import reshard

        if not x.sharding:
            # If input was not sharded (replicated), we gather/replicate the cotangent
            from .all_gather import gather_all_axes

            return gather_all_axes(cotangent)

        return reshard(
            cotangent,
            x.sharding.mesh,
            x.sharding.dim_specs,
            replicated_axes=x.sharding.replicated_axes,
        )

    def infer_sharding_spec(self, args, mesh, kwargs):
        spec = kwargs["spec"]
        input_spec = args[0].sharding
        return spec, [input_spec], False

    def maxpr(
        self,
        x: TensorValue,
        mesh: DeviceMesh,
        dim_specs: list[DimSpec],
        **kwargs: Any,
    ) -> list[TensorValue]:
        """Create sharded TensorValues by slicing the input."""
        from ...core.sharding.spec import ShardingSpec

        global_shape = kwargs.pop("global_shape", None)
        if global_shape is None:
            global_shape = tuple(int(d) for d in x.type.shape)

        spec = ShardingSpec(mesh, dim_specs)

        if "shard_idx" in kwargs:
            shard_idx = kwargs["shard_idx"]
            return self._slice_for_device(x, global_shape, spec, shard_idx, mesh)

        return self._simulate_shard_execution(x, global_shape, spec, mesh)

    def _simulate_shard_execution(self, x, global_shape, spec, mesh):
        """Execute sharding manually for all devices (simulation)."""
        num_shards = len(mesh.devices)
        shard_values = []
        for shard_idx in range(num_shards):
            val = self._slice_for_device(x, global_shape, spec, shard_idx, mesh)

            if mesh.is_distributed:
                val = ops.transfer_to(val, mesh.device_refs[shard_idx])
            shard_values.append(val)

        return shard_values

    def _slice_for_device(self, x, global_shape, spec, shard_idx, mesh):
        from ...core.sharding.spec import compute_local_shape
        from ...core.tensor import Tensor

        effective_x = x
        input_shard_offset = [0] * len(global_shape)

        if isinstance(x, Tensor):
            x.hydrate()
            vals = x._values

            if vals:
                if len(vals) > shard_idx:
                    effective_x = vals[shard_idx]

                    if x.sharding:
                        for d, dim_spec in enumerate(x.sharding.dim_specs):
                            offset = 0
                            shard_pos = 0
                            total_shards = 1
                            for axis in dim_spec.axes:
                                size = mesh.get_axis_size(axis)
                                coord = mesh.get_coordinate(shard_idx, axis)
                                shard_pos = (shard_pos * size) + coord
                                total_shards *= size

                            dim_global_len = int(global_shape[d])
                            chunk_size = math.ceil(dim_global_len / total_shards)
                            input_shard_offset[d] = shard_pos * chunk_size
                else:
                    effective_x = vals[0]

        target_local_shape = compute_local_shape(global_shape, spec, shard_idx)

        slices = []
        for d, (t_len, g_len) in enumerate(
            zip(target_local_shape, global_shape, strict=False)
        ):
            inp_len = int(effective_x.type.shape[d])

            if inp_len == t_len:
                slices.append(slice(0, t_len))
                continue

            dim_spec = spec.dim_specs[d]
            total_shards = 1
            my_shard_pos = 0
            for axis_name in dim_spec.axes:
                size = mesh.get_axis_size(axis_name)
                coord = mesh.get_coordinate(shard_idx, axis_name)
                my_shard_pos = (my_shard_pos * size) + coord
                total_shards *= size

            chunk_size = math.ceil(g_len / total_shards)
            start_global = my_shard_pos * chunk_size
            start_global = min(start_global, g_len)

            end_global = min(start_global + chunk_size, g_len)

            start_local = start_global - input_shard_offset[d]
            end_local = end_global - input_shard_offset[d]

            start_local = max(0, min(start_local, inp_len))
            end_local = max(0, min(end_local, inp_len))

            slices.append(slice(start_local, end_local))

        return effective_x[tuple(slices)]

    def execute(
        self,
        x,
        mesh: DeviceMesh,
        dim_specs: list[DimSpec],
        replicated_axes: set[str] | None = None,
        _bypass_idempotency: bool = False,
    ):
        """Shard a tensor according to the given specification.

        IDEMPOTENT: if the tensor is already sharded with the target spec,
        returns identity to avoid double-execution in shard_map.
        """
        from max import graph as g

        from ...core import GRAPH, Tensor
        from ...core.sharding.spec import ShardingSpec, needs_reshard

        target_spec = ShardingSpec(
            mesh, dim_specs, replicated_axes=replicated_axes or set()
        )

        if not _bypass_idempotency and isinstance(x, Tensor) and x.sharding:
            if not needs_reshard(x.sharding, target_spec):
                return x

            from ...core.sharding.spmd import reshard_tensor

            return reshard_tensor(x, x.sharding, target_spec, mesh)

        global_shape = None
        if isinstance(x, Tensor):
            local = x.physical_local_shape(0)
            if local is not None and x.sharding:
                from ...core.sharding.spec import compute_global_shape

                global_shape = compute_global_shape(tuple(local), x.sharding)
            elif local is not None:
                global_shape = tuple(int(d) for d in local)

        if isinstance(x, Tensor):
            x.hydrate()

        with GRAPH.graph:
            x_input = x
            if isinstance(x, Tensor) and not x._values:
                x_input = g.TensorValue(x)

            shard_values = self.maxpr(
                x_input, mesh, dim_specs, global_shape=global_shape
            )

        spec = ShardingSpec(mesh, dim_specs, replicated_axes=replicated_axes or set())
        output = Tensor._create_unsafe(
            values=shard_values,
            traced=x.traced if isinstance(x, Tensor) else False,
            batch_dims=x.batch_dims if isinstance(x, Tensor) else 0,
        )
        output.sharding = spec

        traced = x.traced if isinstance(x, Tensor) else False

        trace_kwargs = {"mesh": mesh, "dim_specs": dim_specs}
        self._setup_output_refs(output, (x,), trace_kwargs, trace_kwargs, traced)

        return output

    def _compute_global_from_local(self, local_shape, sharding):
        """Deprecated: use spmd.compute_global_shape."""
        from ...core.sharding.spec import compute_global_shape

        return compute_global_shape(local_shape, sharding)


shard_op = ShardOp()


def shard(x, mesh: DeviceMesh, dim_specs: list[DimSpec], **kwargs):
    """Shard a tensor according to the given mesh and dimension specs."""
    return shard_op(x, mesh, dim_specs, **kwargs)
