# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh, ShardingSpec


class ReduceScatterOp(CollectiveOperation):
    """Reduce then scatter the result across shards."""

    @property
    def name(self) -> str:
        return "reduce_scatter"

    @classmethod
    def estimate_cost(
        cls,
        size_bytes: int,
        mesh: DeviceMesh,
        axes: list[str],
        input_specs: list[ShardingSpec] = None,
        output_specs: list[ShardingSpec] = None,
    ) -> float:
        if not axes:
            return 0.0

        n_devices = 1
        for axis in axes:
            n_devices *= mesh.get_axis_size(axis)

        if n_devices <= 1:
            return 0.0

        bandwidth = getattr(mesh, "bandwidth", 1.0)
        cost = (n_devices - 1) / n_devices * size_bytes / bandwidth
        return cost

    def infer_sharding_spec(self, args: Any, mesh: DeviceMesh, kwargs: dict) -> Any:
        """Infer sharding for ReduceScatter (Adaptation Layer)."""
        # CollectiveOperation.infer_sharding_spec handles validation and calls _compute_output_spec
        return super().infer_sharding_spec(args, mesh, kwargs)

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for reduce_scatter: all_gather the gradients."""
        from .all_gather import all_gather

        # Axis is required. Passed as second positional arg by wrapper.
        if isinstance(primals, (list, tuple)) and len(primals) > 1:
            axis = primals[1]
        else:
            axis = 0  # Fallback

        return all_gather(cotangent, axis=axis)

    def execute(self, args: tuple[Any, ...], kwargs: dict) -> Any:
        """Sum-reduce across shards then scatter the result (Physical)."""
        from ...core import GRAPH, Tensor

        sharded_tensor: Tensor = args[0]

        # Handle positional or keyword axis
        if len(args) > 1:
            axis = args[1]
        else:
            axis = kwargs.get("axis")

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

        if mesh and mesh.is_distributed:
            from max.dtype import DType
            from max.graph.ops.allgather import allgather as max_allgather
            from max.graph.type import BufferType

            # Robust implementation via allgather (since native allreduce is broken)
            BUFFER_SIZE = 65536
            signal_buffers = [
                ops.buffer_create(BufferType(DType.uint8, (BUFFER_SIZE,), dev))
                for dev in mesh.device_refs
            ]

            # 1. Gather all inputs (concatenated)
            gathered_list = max_allgather(shard_graph_values, signal_buffers, axis=axis)
            gathered_tensor = gathered_list[0]  # All devices get same data

            # 2. Split into original input chunks
            chunk_shape = shard_graph_values[0].type.shape
            chunk_axis_size = int(chunk_shape[axis])
            num_shards = len(shard_graph_values)

            input_chunks = []
            for i in range(num_shards):
                slices = [slice(None)] * len(chunk_shape)
                slices[axis] = slice(i * chunk_axis_size, (i + 1) * chunk_axis_size)
                input_chunks.append(gathered_tensor[tuple(slices)])

            # 3. Reduce (Sum)
            reduced = input_chunks[0]
            for chunk in input_chunks[1:]:
                reduced = ops.add(reduced, chunk)

            # 4. Scatter (Split result into output chunks)
            # Result shape is same as input chunk shape
            output_axis_size = chunk_axis_size // num_shards

            scattered = []
            for i in range(num_shards):
                slices = [slice(None)] * len(chunk_shape)
                slices[axis] = slice(i * output_axis_size, (i + 1) * output_axis_size)
                scattered.append(reduced[tuple(slices)])

            return scattered

        # Simulation mode: handle grouped execution for multi-axis meshes
        if scatter_axes and mesh and len(mesh.axis_names) > 1:
            # Group shards by their coordinates on axes NOT being scattered
            other_axes = [ax for ax in mesh.axis_names if ax not in scatter_axes]
            if other_axes:
                return self._grouped_scatter(
                    shard_graph_values, axis, mesh, scatter_axes, other_axes
                )

        # Simple case: all shards participate in the same reduce-scatter
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

        for key, group_members in groups.items():
            # Extract the values for this group
            group_shards = [val for _, val in group_members]
            group_indices = [idx for idx, _ in group_members]
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

    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output sharding: the scatter axis becomes sharded on the scatter_axes."""
        from ...core.sharding.spec import DimSpec, ShardingSpec

        mesh = input_tensor.sharding.mesh if input_tensor.sharding else None
        input_spec = input_tensor.sharding

        if mesh and input_spec:
            rank = len(input_spec.dim_specs)
            new_dim_specs = []

            # Use only the axes that are actually being scattered across
            scatter_axes = kwargs.get("scatter_axes", set())
            if not scatter_axes:
                # Fallback: use the axes from the input sharding on the target dim
                kwargs_axis = kwargs.get("axis", 0)
                target_dim = self._get_physical_axis(input_tensor, kwargs_axis)
                if target_dim < len(input_spec.dim_specs):
                    scatter_axes = set(input_spec.dim_specs[target_dim].axes or [])

            kwargs_axis = kwargs.get("axis", 0)
            target_dim = self._get_physical_axis(input_tensor, kwargs_axis)

            for d in range(rank):
                input_d_spec = (
                    input_spec.dim_specs[d] if d < len(input_spec.dim_specs) else None
                )

                if d == target_dim:
                    # Output is sharded on the scatter_axes (same as input sharding on this dim)
                    current_axes = sorted(list(scatter_axes))
                    new_dim_specs.append(DimSpec(current_axes))
                else:
                    new_dim_specs.append(input_d_spec if input_d_spec else DimSpec([]))

            return ShardingSpec(mesh, new_dim_specs)

        return None


reduce_scatter_op = ReduceScatterOp()


def reduce_scatter(sharded_tensor, axis: int, **kwargs):
    """Sum-reduce then scatter result across shards.

    Note: MAX only supports sum reduction natively.
    """
    return reduce_scatter_op(sharded_tensor, axis, **kwargs)
