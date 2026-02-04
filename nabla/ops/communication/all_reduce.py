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


class AllReduceOp(CollectiveOperation):
    """Reduce values across all shards using the specified reduction."""

    @property
    def name(self) -> str:
        return "all_reduce"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for all_reduce."""
        from ...core.sharding import spmd
        from ...core.sharding.spec import compute_local_shape

        x = args[0]
        mesh = self._derive_mesh(x, kwargs) or spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else x.num_shards

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                s = x.shape
            shapes.append(tuple(int(d) for d in s))

        dtypes = [x.dtype] * num_shards

        if mesh:
            if mesh.is_distributed:
                devices = [d for d in mesh.device_refs]
            else:
                devices = [mesh.device_refs[0]] * num_shards
        else:
            devices = [x.device] * (num_shards or 1)

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
        """Estimate AllReduce cost."""
        if not axes:
            return 0.0

        n_devices = 1
        for axis in axes:
            n_devices *= mesh.get_axis_size(axis)

        if n_devices <= 1:
            return 0.0

        bandwidth = getattr(mesh, "bandwidth", 1.0)

        cost = 2.0 * (n_devices - 1) / n_devices * size_bytes / bandwidth
        return cost

    def execute(self, args: tuple[Any, ...], kwargs: dict) -> Any:
        """Sum-reduce across shards (Physical)."""
        from ...core import GRAPH, Tensor

        sharded_tensor: Tensor = args[0]
        reduce_op = kwargs.get("reduce_op", "sum")
        reduce_axes = kwargs.get("reduce_axes")

        # 1. Derive Metadata
        mesh = self._derive_mesh(sharded_tensor, kwargs)
        reduce_axes = self._get_reduce_axes(sharded_tensor, kwargs)

        # 2. Validation & Early Exit
        if not sharded_tensor.sharding:
            return (sharded_tensor.values, sharded_tensor.sharding, None)

        # 3. Execution Context
        with GRAPH.graph:
            values = sharded_tensor.values

            # Handle replicated input simulation
            if mesh and len(values) == 1 and len(mesh.devices) > 1:
                # In simulation, a single value for a distributed mesh implies replication.
                # We expand it to simulate the reduction across the mesh.
                values = [values[0]] * len(mesh.devices)

            # Ported logic from kernel
            reduced_graph_values = self._reduce_logic(
                values, mesh=mesh, reduce_op=reduce_op, reduce_axes=reduce_axes
            )

        # 4. Compute Output Spec
        output_spec = self._compute_output_spec(
            sharded_tensor, reduced_graph_values, reduce_axes=reduce_axes
        )

        return (reduced_graph_values, output_spec, mesh)

    def _reduce_logic(
        self,
        shard_graph_values: list[TensorValue],
        mesh: DeviceMesh = None,
        reduce_op: str = "sum",
        reduce_axes: set[str] = None,
    ) -> list[TensorValue]:
        """Core reduction implementation (MAX ops or simulation)."""
        if not shard_graph_values:
            return []

        # 1. Distributed Execution Path
        if mesh and mesh.is_distributed and len(shard_graph_values) > 1:
            if reduce_op == "sum" and hasattr(ops, "allreduce") and hasattr(ops.allreduce, "sum"):
                return ops.allreduce.sum(shard_graph_values, mesh.get_signal_buffers())

            # Fallback for complex reductions (MAX/MIN/PROD) using native allgather
            from max.graph.ops.allgather import allgather as max_allgather
            gathered = max_allgather(shard_graph_values, mesh.get_signal_buffers(), axis=0)

            result_graph_values = []
            num_shards = len(shard_graph_values)
            chunk_size = shard_graph_values[0].type.shape[0]

            for gathered_tensor in gathered:
                chunks = [gathered_tensor[i * chunk_size : (i + 1) * chunk_size] for i in range(num_shards)]
                
                reduced = chunks[0]
                for chunk in chunks[1:]:
                    if reduce_op == "sum": reduced = ops.add(reduced, chunk)
                    elif reduce_op == "max": reduced = ops.max(reduced, chunk)
                    elif reduce_op == "min": reduced = ops.min(reduced, chunk)
                    elif reduce_op == "prod": reduced = ops.mul(reduced, chunk)
                    else: raise ValueError(f"Unknown reduction op: {reduce_op}")
                result_graph_values.append(reduced)

            return result_graph_values

        # 2. CPU Simulation Path (Local execution)
        if reduce_axes and mesh and not mesh.is_distributed:
            return self.simulate_grouped_execution(
                shard_graph_values, mesh, reduce_axes, reduce_op=reduce_op
            )

        # 3. Simple Fallback (Single shard or generic)
        result = shard_graph_values[0]
        for sv in shard_graph_values[1:]:
            if reduce_op == "sum": result = ops.add(result, sv)
            elif reduce_op == "max": result = ops.max(result, sv)
            elif reduce_op == "min": result = ops.min(result, sv)
            elif reduce_op == "prod": result = ops.mul(result, sv)
            else: raise ValueError(f"Unknown reduction op: {reduce_op}")

        return [result] * len(shard_graph_values)

    def infer_sharding_spec(self, args: Any, mesh: DeviceMesh, kwargs: dict) -> Any:
        """Infer sharding for AllReduce (Adaptation Layer)."""
        input_tensor = args[0]
        input_sharding = input_tensor.sharding
        output_sharding = self._compute_output_spec(input_tensor, None, **kwargs)
        return output_sharding, [input_sharding], False

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for AllReduce (sum): assign replicated gradient to shards."""
        input_tensor = primals[0] if isinstance(primals, (list, tuple)) else primals

        if not input_tensor.sharding:
            return cotangent

        # VJP for all_reduce (sum) propagates the gradient to all inputs.
        # Since all_reduce acts as a sum reduction over shards, the gradient w.r.t
        # each input shard is equal to the gradient w.r.t the output shard
        # (which acts as the sum).
        # We need to construct a tensor with the INPUT's sharding spec,
        # but with the COTANGENT's values as the physical content.
        # We bypass reshard() because reshard() would attempt to slice the cotangent
        # based on global shape, but here the cotangent ALREADY represents the
        # correct physical content for the input shards.

        from .reshard import reshard

        return reshard(
            cotangent,
            input_tensor.sharding.mesh,
            input_tensor.sharding.dim_specs,
            replicated_axes=input_tensor.sharding.replicated_axes,
            global_shape=input_tensor.physical_global_shape,
        )

    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output clears partial flags but preserves axes mappings for non-partial dims."""
        from ...core.sharding.spec import ShardingSpec, DimSpec

        if not input_tensor.sharding:
            return None

        reduce_axes = kwargs.get("reduce_axes")
        if isinstance(reduce_axes, str):
            reduce_axes = {reduce_axes}
        elif isinstance(reduce_axes, (list, tuple)):
            reduce_axes = set(reduce_axes)

        if reduce_axes is None:
            # Full reduction over all sharding axes -> Output is fully replicated
            new_dim_specs = [DimSpec([]) for _ in input_tensor.sharding.dim_specs]
            return ShardingSpec(
                input_tensor.sharding.mesh, new_dim_specs, partial_sum_axes=set()
            )

        new_spec = input_tensor.sharding.clone()
        new_spec.partial_sum_axes = set(
            ax for ax in new_spec.partial_sum_axes if ax not in reduce_axes
        )

        for ds in new_spec.dim_specs:
            ds.axes = tuple(ax for ax in ds.axes if ax not in reduce_axes)
            if not ds.axes:
                ds.partial = False

        return new_spec

    def simulate_grouped_execution(
        self,
        shard_results: list[TensorValue],
        mesh: DeviceMesh,
        reduce_axes: set[str],
        reduce_op: str = "sum",
    ) -> list[TensorValue]:
        """Simulate grouped AllReduce execution for SPMD verification."""
        if not reduce_axes:
            return shard_results

        num_shards = len(shard_results)

        all_axes = set(mesh.axis_names)
        if all_axes.issubset(reduce_axes):
            return self._reduce_logic(shard_results, mesh=mesh, reduce_op=reduce_op)

        group_axes = [ax for ax in all_axes if ax not in reduce_axes]
        groups = self._group_shards_by_axes(shard_results, mesh, group_axes)

        new_results = [None] * num_shards

        for key, group_members in groups.items():

            group_shards = [val for _, val in group_members]

            if len(group_shards) > 1:
                curr_reduced = self._reduce_logic(
                    group_shards, mesh=mesh, reduce_op=reduce_op
                )
            else:
                curr_reduced = [group_shards[0]]

            for i, (shard_idx, _) in enumerate(group_members):
                new_results[shard_idx] = (
                    curr_reduced[i] if isinstance(curr_reduced, list) else curr_reduced
                )

        return new_results


class PMeanOp(CollectiveOperation):
    """Compute mean across all shards (psum / axis_size)."""

    @property
    def name(self) -> str:
        return "pmean"

    def compute_physical_shape(
        self, args: tuple, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for pmean (local shape preserved)."""
        from ...core.sharding import spmd

        x = args[0]
        mesh = self._derive_mesh(x, kwargs) or spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else x.num_shards

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is None:
                s = x.shape
            shapes.append(tuple(int(d) for d in s))

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
        """Compute mean across shards (Physical)."""
        from ...core import GRAPH

        # 1. Perform AllReduce first
        shard_graph_values, output_spec, mesh = all_reduce_op.execute(args, kwargs)

        axis_name = kwargs.get("axis_name")

        # 2. Rescale
        with GRAPH.graph:
            if axis_name and mesh:
                axis_size = mesh.get_axis_size(axis_name)
            else:
                axis_size = len(shard_graph_values)

            dtype = shard_graph_values[0].type.dtype
            device = shard_graph_values[0].type.device
            scale = ops.constant(1.0 / axis_size, dtype, device)
            scaled_graph_values = [ops.mul(r, scale) for r in shard_graph_values]

        return (scaled_graph_values, output_spec, mesh)

    def _compute_output_spec(self, input_tensor, results, **kwargs):
        """Output clears partial flags but preserves axes mappings for non-partial dims."""
        from ...core.sharding.spec import ShardingSpec

        if not input_tensor.sharding:
            return None

        new_spec = input_tensor.sharding.clone()
        new_spec.partial_sum_axes.clear()
        for ds in new_spec.dim_specs:
            ds.partial = False

        return new_spec


all_reduce_op = AllReduceOp()
pmean_op = PMeanOp()


def all_reduce(sharded_tensor, **kwargs):
    """Sum-reduce across all shards.

    Note: MAX only supports sum reduction natively.
    """
    return all_reduce_op(sharded_tensor, **kwargs)


def pmean(sharded_tensor, axis_name: str = None):
    """Compute mean across all shards.

    Equivalent to psum(x) / axis_size.
    """
    return pmean_op(sharded_tensor, axis_name=axis_name)


__all__ = ["AllReduceOp", "all_reduce", "pmean"]
