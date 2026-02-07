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

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for all_to_all (split then concat)."""
        from ...core.sharding import spmd

        x = args[0]
        split_axis = kwargs.get("split_axis", 0)
        concat_axis = kwargs.get("concat_axis", 0)

        mesh = self._derive_mesh(x, kwargs) or spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else x.num_shards

        phys_split_axis = self._get_physical_axis(x, split_axis)
        phys_concat_axis = self._get_physical_axis(x, concat_axis)

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                s = x.shape

            if s is None:
                s = x.shape

            out_shape = list(int(d) for d in s)
            if phys_split_axis != phys_concat_axis:
                if out_shape[phys_split_axis] % num_shards != 0:
                    raise ValueError(
                        f"all_to_all split axis size {out_shape[phys_split_axis]} not divisible by {num_shards}"
                    )
                out_shape[phys_split_axis] //= num_shards
                out_shape[phys_concat_axis] *= num_shards

            shapes.append(tuple(out_shape))

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    # _get_shifted_axes helper removed in favor of centralized _get_physical_axis

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for all_to_all: another all_to_all with swapped axes."""
        split_axis = output.op_kwargs.get("split_axis")
        concat_axis = output.op_kwargs.get("concat_axis")
        from .all_to_all import all_to_all

        # Swap split and concat for backward
        return all_to_all(cotangent, split_axis=concat_axis, concat_axis=split_axis)

    def execute(self, args: tuple[Any, ...], kwargs: dict) -> Any:
        """All-to-all distributed transpose (Physical)."""
        from ...core import GRAPH, Tensor

        sharded_tensor: Tensor = args[0]
        tiled = kwargs.get("tiled", True)

        split_axis = kwargs.get("split_axis", 0)
        concat_axis = kwargs.get("concat_axis", 0)

        # 1. Derive Metadata
        mesh = self._derive_mesh(sharded_tensor, kwargs)

        # Calculate physical axes
        phys_split_axis = self._get_physical_axis(sharded_tensor, split_axis)
        phys_concat_axis = self._get_physical_axis(sharded_tensor, concat_axis)

        # 2. Validation & Early Exit
        if not sharded_tensor.sharding:
            return (sharded_tensor.values, None, None)

        # 3. Execution Context
        with GRAPH.graph:
            values = sharded_tensor.values

            # Ported logic from kernel
            result_graph_values = self._all_to_all_logic(
                values, phys_split_axis, phys_concat_axis, mesh=mesh, tiled=tiled
            )

        # 4. Compute Output Spec
        output_spec = self._compute_output_spec(
            sharded_tensor,
            result_graph_values,
            split_axis=split_axis,
            concat_axis=concat_axis,
        )

        return (result_graph_values, output_spec, mesh)

    def _all_to_all_logic(
        self,
        shard_graph_values: list[TensorValue],
        split_axis: int,
        concat_axis: int,
        mesh: DeviceMesh = None,
        tiled: bool = True,
    ) -> list[TensorValue]:
        """All-to-all: distributed transpose of tensor blocks."""
        num_devices = len(shard_graph_values)

        if num_devices <= 1:
            return shard_graph_values

        chunks_per_device = []
        for val in shard_graph_values:
            shape = val.type.shape
            axis_size = int(shape[split_axis])
            chunk_size = axis_size // num_devices

            if axis_size % num_devices != 0:
                raise ValueError(
                    f"Split axis size {axis_size} not divisible by {num_devices} devices"
                )

            # Split each shard into chunks for every destination device
            chunks = [val[tuple(slice(i*chunk_size, (i+1)*chunk_size) if d == split_axis else slice(None) 
                      for d in range(len(shape)))] for i in range(num_devices)]
            chunks_per_device.append(chunks)

        # Transpose communication: send chunks to their respective devices
        results = []
        for dst in range(num_devices):
            received = []
            for src in range(num_devices):
                chunk = chunks_per_device[src][dst]
                if mesh and mesh.is_distributed:
                    chunk = ops.transfer_to(chunk, mesh.device_refs[dst])
                received.append(chunk)
            
            # Reassemble on destination
            results.append(ops.concat(received, axis=concat_axis) if tiled else ops.stack(received, axis=concat_axis))

        return results

    def _compute_output_spec(self, input_tensor, results, input_sharding=None, **kwargs):
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

        input_sharding = input_sharding or input_tensor.sharding
        mesh = input_sharding.mesh if input_sharding else None

        if mesh and input_sharding:
            split_axis = kwargs.get("split_axis", 0)
            concat_axis = kwargs.get("concat_axis", 0)

            phys_split_axis = self._get_physical_axis(input_tensor, split_axis)
            phys_concat_axis = self._get_physical_axis(input_tensor, concat_axis)

            new_dim_specs = [
                DimSpec(list(ds.axes), is_open=ds.is_open)
                for ds in input_sharding.dim_specs
            ]

            # If sharding mismatch, just return None or raise. AllToAll assumes valid input spec.
            if phys_concat_axis >= len(new_dim_specs) or phys_split_axis >= len(
                new_dim_specs
            ):
                # This might happen if shapes are weird, but usually protected by validation
                return None

            source_axes = new_dim_specs[phys_concat_axis].axes
            target_axes = new_dim_specs[phys_split_axis].axes

            if source_axes:
                moved_axes = list(source_axes)
                new_dim_specs[phys_concat_axis] = DimSpec([], is_open=True)
                new_dim_specs[phys_split_axis] = DimSpec(
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
