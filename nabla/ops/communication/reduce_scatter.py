# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from ..base import OpArgs, OpKwargs, OpResult
from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, ShardingSpec


class ReduceScatterOp(CollectiveOperation):
    """Reduce then scatter the result across shards."""

    @property
    def name(self) -> str:
        return "reduce_scatter"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for reduce_scatter (reduce then split axis)."""
        from ...core.sharding import spmd

        x = args[0]
        axis = kwargs.get("axis")
        if axis is None and len(args) > 1:
            axis = args[1]
        if axis is None:
            axis = 0

        mesh = self._derive_mesh(x, kwargs) or spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else x.num_shards

        physical_axis = self._get_physical_axis(x, axis)

        scatter_axes = set()
        if x.sharding and physical_axis < len(x.sharding.dim_specs):
            scatter_axes = set(x.sharding.dim_specs[physical_axis].axes or [])
        if not scatter_axes and mesh:
            scatter_axes = set(mesh.axis_names)

        scatter_factor = 1
        if mesh and scatter_axes:
            for ax in scatter_axes:
                scatter_factor *= mesh.get_axis_size(ax)

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                s = x.shape

            out_shape = [int(d) for d in s]
            if scatter_factor > 1 and 0 <= physical_axis < len(out_shape):
                if out_shape[physical_axis] % scatter_factor != 0:
                    raise ValueError(
                        f"reduce_scatter axis size {out_shape[physical_axis]} not divisible by {scatter_factor}"
                    )
                out_shape[physical_axis] //= scatter_factor

            shapes.append(tuple(out_shape))

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

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

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for reduce_scatter: all_gather the gradients."""
        from .all_gather import all_gather

        # Axis is required. Passed as second positional arg by wrapper.
        axis = primals[1] if len(primals) > 1 else 0  # Fallback

        return [all_gather(cotangents[0], axis=axis)]

    def execute(
        self, args: OpArgs, kwargs: OpKwargs
    ) -> tuple[list[TensorValue], ShardingSpec | None, DeviceMesh | None]:
        """Sum-reduce across shards then scatter the result (Physical)."""
        from ...core import GRAPH, Tensor

        sharded_tensor: Tensor = args[0]

        # Handle positional or keyword axis
        axis = args[1] if len(args) > 1 else kwargs.get("axis")

        if axis is None:
            raise ValueError("ReduceScatterOp requires an 'axis' argument.")

        # 1. Derive Metadata
        mesh = self._derive_mesh(sharded_tensor, kwargs)

        # Calculate physical axis
        physical_axis = self._get_physical_axis(sharded_tensor, axis)

        # 2. Validation & Early Exit
        if not sharded_tensor.sharding:
            return (sharded_tensor.values, None, None)

        # Get the mesh axes that the scatter dimension is sharded on
        input_spec = sharded_tensor.sharding
        scatter_axes = set()
        if physical_axis < len(input_spec.dim_specs):
            scatter_axes = set(input_spec.dim_specs[physical_axis].axes or [])

        # If scatter_axes is empty (input was replicated on this axis), scatter across ALL mesh axes
        if not scatter_axes and mesh:
            scatter_axes = set(mesh.axis_names)

        # 2. Execution Context
        with GRAPH.graph:
            values = sharded_tensor.values

            # Ported logic from kernel
            scattered_graph_values = self._scatter_logic(
                values, physical_axis, mesh=mesh, scatter_axes=scatter_axes
            )

        # 3. Compute Output Spec
        # Use logical axis; _compute_output_spec converts it.
        output_spec = self._compute_output_spec(
            sharded_tensor, scattered_graph_values, axis=axis, scatter_axes=scatter_axes
        )

        return (scattered_graph_values, output_spec, mesh)

    def _scatter_logic(
        self,
        shard_graph_values: list[TensorValue],
        axis: int,
        mesh: DeviceMesh = None,
        scatter_axes: set[str] = None,
    ) -> list[TensorValue]:
        """Core reduce-scatter implementation (MAX ops or simulation)."""

        # 1. Distributed Execution Path
        if mesh and mesh.is_distributed:
            from max.dtype import DType
            from max.graph.type import BufferType

            signal_buffers = [
                ops.buffer_create(BufferType(DType.uint8, (65536,), dev))
                for dev in mesh.device_refs
            ]

            if hasattr(ops, "reducescatter") and hasattr(ops.reducescatter, "sum"):
                return ops.reducescatter.sum(
                    shard_graph_values, signal_buffers, axis=axis
                )

            # Robust fallback via native allgather:
            # 1. Every device gets all chunks.
            # 2. Every device reduces all chunks locally.
            # 3. Every device slices its own disjoint result portion.
            from max.graph.ops.allgather import allgather as max_allgather

            gathered_results = max_allgather(
                shard_graph_values, signal_buffers, axis=axis
            )

            num_shards = len(shard_graph_values)
            chunk_shape = shard_graph_values[0].type.shape
            chunk_axis_size = int(chunk_shape[axis])

            def _slice_axis(tensor, start, end):
                slices = [slice(None)] * len(chunk_shape)
                slices[axis] = slice(start, end)
                return tensor[tuple(slices)]

            results = []
            for device_idx, gathered in enumerate(gathered_results):
                # Local reduction of all chunks
                reduced = _slice_axis(gathered, 0, chunk_axis_size)
                for i in range(1, num_shards):
                    chunk = _slice_axis(
                        gathered, i * chunk_axis_size, (i + 1) * chunk_axis_size
                    )
                    reduced = ops.add(reduced, chunk)

                # Pick the device-specific portion of the reduced result
                out_size = chunk_axis_size // num_shards
                results.append(
                    _slice_axis(
                        reduced, device_idx * out_size, (device_idx + 1) * out_size
                    )
                )

            return results

        # 2. CPU Simulation Path (Grouped/Sharded execution)
        if scatter_axes and mesh and len(mesh.axis_names) > 1:
            # Group shards by their coordinates on axes NOT being scattered
            other_axes = [ax for ax in mesh.axis_names if ax not in scatter_axes]
            if other_axes:
                return self._grouped_scatter(
                    shard_graph_values, axis, mesh, scatter_axes, other_axes
                )

        # Simple case: all shards participate in the same reduce-scatter
        # Handle replicated input (single value for multiple shards)
        if len(shard_graph_values) == 1 and mesh and len(mesh.devices) > 1:
            # Expand for simulation
            shard_graph_values = [shard_graph_values[0]] * len(mesh.devices)

        full_result = shard_graph_values[0]
        for sv in shard_graph_values[1:]:
            full_result = ops.add(full_result, sv)

        num_shards = len(shard_graph_values)
        shape = full_result.type.shape
        axis_size = int(shape[axis])
        chunk_size = axis_size // num_shards

        scattered = []
        for i in range(num_shards):
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(i * chunk_size, (i + 1) * chunk_size)
            scattered.append(full_result[tuple(slices)])

        return scattered

    def _grouped_scatter(
        self,
        shard_graph_values: list[TensorValue],
        axis: int,
        mesh: DeviceMesh,
        scatter_axes: set[str],
        other_axes: list[str],
    ) -> list[TensorValue]:
        """Perform reduce-scatter within groups defined by non-scatter axes.

        This handles multi-axis meshes where we only scatter along specific axes.
        Shards with the same coordinates on 'other_axes' form a group and
        participate in the same reduce-scatter operation.
        """
        # Group shards by their coordinates on the other axes
        groups = self._group_shards_by_axes(shard_graph_values, mesh, other_axes)

        num_total_shards = len(shard_graph_values)
        new_results = [None] * num_total_shards

        for _key, group_members in groups.items():
            # Extract the values for this group
            group_shards = [val for _, val in group_members]
            _group_indices = [idx for idx, _ in group_members]
            num_in_group = len(group_shards)

            if num_in_group <= 1:
                # Only one shard in group - nothing to reduce or scatter
                for shard_idx, val in group_members:
                    new_results[shard_idx] = val
                continue

            # Reduce within the group
            full_result = group_shards[0]
            for sv in group_shards[1:]:
                full_result = ops.add(full_result, sv)

            # Scatter: each shard in the group gets a portion
            shape = full_result.type.shape
            axis_size = int(shape[axis])
            chunk_size = axis_size // num_in_group

            for i, (shard_idx, _) in enumerate(group_members):
                slices = [slice(None)] * len(shape)
                slices[axis] = slice(i * chunk_size, (i + 1) * chunk_size)
                new_results[shard_idx] = full_result[tuple(slices)]

        return new_results

    def _compute_output_spec(
        self, input_tensor, results, input_sharding=None, **kwargs
    ):
        """Output sharding: the scatter axis becomes sharded on the scatter_axes."""
        from ...core.sharding.spec import DimSpec, ShardingSpec

        input_sharding = input_sharding or input_tensor.sharding
        mesh = input_sharding.mesh if input_sharding else None

        if mesh and input_sharding:
            rank = len(input_sharding.dim_specs)
            new_dim_specs = []

            # Use only the axes that are actually being scattered across
            scatter_axes = kwargs.get("scatter_axes", set())
            if not scatter_axes:
                # Fallback: use the axes from the input sharding on the target dim
                kwargs_axis = kwargs.get("axis", 0)
                target_dim = self._get_physical_axis(input_tensor, kwargs_axis)
                if target_dim < len(input_sharding.dim_specs):
                    scatter_axes = set(input_sharding.dim_specs[target_dim].axes or [])

            # If still no scatter axes (input was replicated), scatter across ALL mesh axes
            if not scatter_axes and mesh:
                scatter_axes = set(mesh.axis_names)

            kwargs_axis = kwargs.get("axis", 0)
            target_dim = self._get_physical_axis(input_tensor, kwargs_axis)

            for d in range(rank):
                input_d_spec = (
                    input_sharding.dim_specs[d]
                    if d < len(input_sharding.dim_specs)
                    else None
                )

                if d == target_dim:
                    # Output is sharded on the scatter_axes (same as input sharding on this dim)
                    current_axes = sorted(scatter_axes)
                    new_dim_specs.append(DimSpec(current_axes))
                else:
                    new_dim_specs.append(input_d_spec if input_d_spec else DimSpec([]))

            return ShardingSpec(mesh, new_dim_specs)

        return None


_reduce_scatter_op = ReduceScatterOp()


def reduce_scatter(sharded_tensor, axis: int, **kwargs):
    """Sum-reduce then scatter result across shards.

    Note: MAX only supports sum reduction natively.
    """
    return _reduce_scatter_op([sharded_tensor], {"axis": axis, **kwargs})[0]
