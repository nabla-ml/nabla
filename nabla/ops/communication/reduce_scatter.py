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
            axis = 0 # Fallback 
            
        return all_gather(cotangent, axis=axis)

    def physical_execute(self, args: tuple[Any, ...], kwargs: dict) -> Any:
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

        # 2. Execution Context
        with GRAPH.graph:
            values = sharded_tensor.values
            
            # Ported logic from maxpr
            scattered_values = self._scatter_logic(
                values, physical_axis, mesh=mesh
            )

        # 3. Compute Output Spec
        # Use logical axis; _compute_output_spec converts it.
        output_spec = self._compute_output_spec(
            sharded_tensor, scattered_values, axis=axis
        )

        return (scattered_values, output_spec, mesh)

    def _scatter_logic(
        self,
        shard_values: list[TensorValue],
        axis: int,
        mesh: DeviceMesh = None,
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
            gathered_list = max_allgather(shard_values, signal_buffers, axis=axis)
            gathered_tensor = gathered_list[0]  # All devices get same data

            # 2. Split into original input chunks
            chunk_shape = shard_values[0].type.shape
            chunk_axis_size = int(chunk_shape[axis])
            num_shards = len(shard_values)

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

        full_result = shard_values[0]
        for sv in shard_values[1:]:
            full_result = ops.add(full_result, sv)

        num_shards = len(shard_values)
        shape = full_result.type.shape
        axis_size = int(shape[axis])
        chunk_size = axis_size // num_shards

        scattered = []
        for i in range(num_shards):

            slices = [slice(None)] * len(shape)
            slices[axis] = slice(i * chunk_size, (i + 1) * chunk_size)
            scattered.append(full_result[tuple(slices)])

        return scattered

    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output sharding: the scatter axis becomes sharded."""
        from ...core.sharding.spec import DimSpec, ShardingSpec

        mesh = input_tensor.sharding.mesh if input_tensor.sharding else None
        input_spec = input_tensor.sharding

        if mesh and input_spec:
            rank = len(input_spec.dim_specs)
            new_dim_specs = []

            mesh_axes = mesh.axis_names
            # target_mesh_axis = mesh_axes[0] if mesh_axes else "unknown" # OLD BUGGY LOGIC

            kwargs_axis = kwargs.get("axis", 0)
            target_dim = self._get_physical_axis(input_tensor, kwargs_axis)

            for d in range(rank):
                input_d_spec = (
                    input_spec.dim_specs[d] if d < len(input_spec.dim_specs) else None
                )

                if d == target_dim:
                    current_axes = sorted(
                        list(
                            set(input_d_spec.axes if input_d_spec else [])
                            | set(mesh_axes) # Add ALL mesh axes
                        )
                    )
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
