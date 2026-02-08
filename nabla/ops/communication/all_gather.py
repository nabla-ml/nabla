# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for all_gather (gather along axis)."""
        from ...core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis")
        physical_axis = kwargs.get("physical_axis")
        if physical_axis is None and axis is not None:
            physical_axis = self._get_physical_axis(x, axis)

        mesh = self._derive_mesh(x, kwargs) or spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else x.num_shards

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                s = x.shape

            out_shape = list(int(d) for d in s)
            if (
                x.sharding
                and physical_axis is not None
                and 0 <= physical_axis < len(x.sharding.dim_specs)
            ):
                dim_spec = x.sharding.dim_specs[physical_axis]
                if dim_spec.axes and mesh:
                    axis_factor = 1
                    for ax in dim_spec.axes:
                        axis_factor *= mesh.get_axis_size(ax)
                    out_shape[physical_axis] *= axis_factor

            shapes.append(tuple(out_shape))

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def vjp_rule(self, primals: list, cotangents: list, outputs: list, kwargs: dict) -> list:
        """VJP for all_gather: reshard back to input's sharding."""
        x = primals[0]
        from .reshard import reshard

        if not x.sharding:
            return [cotangents[0]]

        return [reshard(
            cotangents[0],
            x.sharding.mesh,
            x.sharding.dim_specs,
            replicated_axes=x.sharding.replicated_axes,
        )]

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: DeviceMesh,
        axes: list[str],
        input_specs: list[ShardingSpec] = None,
        output_specs: list[ShardingSpec] = None,
    ) -> float:
        return CollectiveOperation._ring_cost(size_bytes, mesh, axes)

    def _compute_output_spec(self, input_tensor, results, input_sharding=None, **kwargs):
        """Compute output sharding spec for AllGather."""
        input_sharding = input_sharding or input_tensor.sharding
        if not input_sharding:
            return None

        # Helper handles logical->physical mapping including vmap batch dims
        physical_axis = kwargs.get("physical_axis")
        if physical_axis is None:
            axis = kwargs.get("axis")
            if axis is not None:
                physical_axis = self._get_physical_axis(input_tensor, axis)

        from ...core.sharding.spec import DimSpec, ShardingSpec

        new_dim_specs = list(input_sharding.dim_specs)

        if physical_axis is not None and 0 <= physical_axis < len(new_dim_specs):
            new_dim_specs[physical_axis] = DimSpec([])  # Replicated

        return ShardingSpec(input_sharding.mesh, new_dim_specs)

    def execute(self, args: list, kwargs: dict) -> Any:
        """Gather shards along an axis to produce replicated full tensors (Physical).

        Derives all physical context (mesh, sharded_axis_name) internally.
        """
        from ...core import GRAPH, Tensor
        from ...core.sharding.spec import DimSpec, ShardingSpec

        sharded_tensor: Tensor = args[0]

        # Handle positional or keyword axis
        axis = None
        if len(args) > 1:
            axis = args[1]
        else:
            axis = kwargs.get("axis")

        physical_axis = kwargs.get("physical_axis")

        if axis is None and physical_axis is None:
            raise ValueError(
                "AllGatherOp requires an 'axis' or 'physical_axis' argument."
            )

        # 1. Derive Metadata
        mesh = self._derive_mesh(sharded_tensor, kwargs)

        # Calculate physical axis for execution logic and output spec
        if physical_axis is None:
            physical_axis = self._get_physical_axis(sharded_tensor, axis)

        # Derive sharded_axis_name
        sharded_axis_name = None
        if sharded_tensor.sharding and 0 <= physical_axis < len(
            sharded_tensor.sharding.dim_specs
        ):
            dim_spec = sharded_tensor.sharding.dim_specs[physical_axis]
            if dim_spec.axes:
                sharded_axis_name = dim_spec.axes[0]

        # 2. Validation & Early Exit
        if not sharded_axis_name:
            return (sharded_tensor.values, sharded_tensor.sharding, mesh)

        # 3. Execution Context
        with GRAPH.graph:
            values = sharded_tensor.values
            gathered_graph_values = self._gather_logic(
                values, physical_axis, mesh, sharded_axis_name
            )

        # 4. Compute Output Spec
        # Now we can safely use the centralized logic.
        output_spec = self._compute_output_spec(
            sharded_tensor,
            gathered_graph_values,
            axis=axis,
            physical_axis=physical_axis,
        )

        return (gathered_graph_values, output_spec, mesh)

    def _gather_logic(
        self,
        shard_graph_values: list[TensorValue],
        axis: int,
        mesh: DeviceMesh = None,
        sharded_axis_name: str = None,
    ) -> list[TensorValue]:
        """Core gather implementation (MAX ops or simulation)."""

        # 1. Distributed Execution Path
        if mesh and mesh.is_distributed:
            from max.graph.ops.allgather import allgather as max_allgather
            return max_allgather(shard_graph_values, mesh.get_signal_buffers(), axis=axis)

        # 2. CPU Simulation Path (Local execution)
        if mesh is None or (sharded_axis_name is None and len(mesh.axis_names) <= 1):
            if len(shard_graph_values) <= 1:
                return (
                    [shard_graph_values[0]] * len(shard_graph_values)
                    if shard_graph_values
                    else []
                )
            full_tensor = ops.concat(shard_graph_values, axis=axis)
            return [full_tensor] * len(shard_graph_values)

        return self._simulate_grouped_gather(
            shard_graph_values, axis, mesh, sharded_axis_name
        )

    def _simulate_grouped_gather(
        self, shard_graph_values, axis, mesh, sharded_axis_name
    ):
        """Handle simulated gather logic for multi-dimensional meshes."""

        all_axes = list(mesh.axis_names)
        other_axes = [ax for ax in all_axes if ax != sharded_axis_name]

        if not other_axes:

            unique_shards = []
            seen_coords = set()
            for shard_idx, val in enumerate(shard_graph_values):
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
            return [full_tensor] * len(shard_graph_values)

        groups = self._group_shards_by_axes(shard_graph_values, mesh, other_axes)

        gathered_per_group = {}
        for other_coords, members in groups.items():

            members.sort(key=lambda x: mesh.get_coordinate(x[0], sharded_axis_name))
            shards = [val for _, val in members]
            gathered = ops.concat(shards, axis=axis) if len(shards) > 1 else shards[0]
            gathered_per_group[other_coords] = gathered

        results = []
        for shard_idx in range(len(shard_graph_values)):
            other_coords = tuple(
                mesh.get_coordinate(shard_idx, ax) for ax in other_axes
            )
            results.append(gathered_per_group[other_coords])
        return results


class GatherAllAxesOp(Operation):
    """Gather all sharded axes to produce a fully replicated tensor.

    Uses hierarchical concatenation to properly handle multi-axis sharding.
    This is the correct way to reconstruct a tensor sharded on multiple dimensions.
    """

    @property
    def name(self) -> str:
        return "gather_all_axes"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for gather_all_axes (full global shape)."""
        from ...core.sharding import spmd

        x = args[0]
        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        # Determine global physical shape (handles uneven shards if present)
        if hasattr(x, "physical_global_shape") and x.physical_global_shape is not None:
            global_shape = tuple(int(d) for d in x.physical_global_shape)
        else:
            local = x.physical_local_shape(0)
            if local is None:
                raise RuntimeError(
                    f"Could not determine physical shape for {self.name}"
                )
        shapes = [tuple(int(d) for d in global_shape)] * num_shards

        dtypes = [x.dtype] * num_shards
        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.devices]
            else:
                devices = [mesh.devices[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

        return shapes, dtypes, devices

    def vjp_rule(self, primals: list, cotangents: list, outputs: list, kwargs: dict) -> list:
        """VJP for gather_all_axes: reshard back to input's sharding."""
        x = primals[0]
        from .reshard import reshard

        if not x.sharding:
            return [cotangents[0]]

        return [reshard(
            cotangents[0],
            x.sharding.mesh,
            x.sharding.dim_specs,
            replicated_axes=x.sharding.replicated_axes,
        )]

    def infer_sharding_spec(self, args: Any, mesh: Any, kwargs: dict) -> Any:
        """Infer sharding: Input preserves current sharding, Output is replicated."""
        if not args:
            return None, [], False

        x = args[0]
        input_shardings = [
            x.sharding if hasattr(x, "sharding") and x.sharding else None
        ]

        # Output is fully replicated
        rank = len(x.shape)
        from ...core.sharding.spmd import create_replicated_spec

        output_sharding = create_replicated_spec(mesh, rank)

        return output_sharding, input_shardings, False

    def execute(self, args: list, kwargs: dict) -> Any:
        """Physical execution for GatherAllAxesOp."""
        from ...core import GRAPH, Tensor
        from ...core.sharding.spmd import create_replicated_spec

        sharded_tensor: Tensor = args[0]
        # Derive mesh from sharding metadata
        mesh = sharded_tensor.sharding.mesh if sharded_tensor.sharding else None

        if not sharded_tensor.sharding:
            return (sharded_tensor.values, None, mesh)

        with GRAPH.graph:
            gathered_shard = self._reconstruct_global_tensor(
                sharded_tensor.values, sharded_tensor.sharding, mesh
            )
            num_shards = len(mesh.devices) if mesh else 1
            results = [gathered_shard] * num_shards

        rank = len(sharded_tensor.sharding.dim_specs)
        output_spec = create_replicated_spec(mesh, rank)
        return (results, output_spec, mesh)

    def _reconstruct_global_tensor(
        self,
        shard_graph_values: list[TensorValue],
        source_spec: ShardingSpec,
        mesh: DeviceMesh = None,
    ) -> TensorValue:
        """Reconstruct the global tensor from potentially multi-dimensional shards."""
        if source_spec.is_fully_replicated():
            return shard_graph_values[0]

        current_shard_descs = [
            (shard_graph_values[i], source_spec.mesh.devices[i])
            for i in range(len(shard_graph_values))
        ]
        rank = len(shard_graph_values[0].type.shape)

        current_active_axes = set()
        for dim in source_spec.dim_specs:
            current_active_axes.update(dim.axes)
        current_active_axes.update(source_spec.replicated_axes)

        # print(f"DEBUG: Reconstruct Start. Rank={rank}. Shards={len(shard_graph_values)}. Spec={source_spec.dim_specs} Active={current_active_axes} MeshAxes={source_spec.mesh.axis_names}")

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

                        c = source_spec.mesh.get_coordinate(device_id, check_ax)
                        signature.append((check_ax, c))

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


_all_gather_op = AllGatherOp()
_gather_all_axes_op = GatherAllAxesOp()


def all_gather(sharded_tensor, axis: int = None, **kwargs):
    """Gather all shards to produce a replicated tensor."""
    return _all_gather_op([sharded_tensor], {"axis": axis, **kwargs})[0]


def gather_all_axes(sharded_tensor):
    """Gather all sharded axes to produce a fully replicated tensor."""
    return _gather_all_axes_op([sharded_tensor], {})[0]
