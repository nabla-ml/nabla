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

    def _should_proceed(self, tensor):
        """Check if all_reduce should proceed."""

        has_multiple_shards = (tensor._values and len(tensor._values) > 1) or (
            tensor._storages and len(tensor._storages) > 1
        )
        if not has_multiple_shards:
            return False

        if tensor.sharding and tensor.sharding.is_fully_replicated():
            return False

        return True

    def adapt_kwargs(self, args: tuple, kwargs: dict, max_batch_dims: int) -> dict:
        """Recover reduce_axes for multi-dim mesh rehydration."""
        if "reduce_axes" in kwargs:
            return kwargs
        
        new_kwargs = dict(kwargs)
        sharded_tensor = args[0]
        
        if sharded_tensor.sharding:
             # If we're performing all_reduce during backprop, it's often to reduce
             # partial axes. We can find them in the GRADIENT'S sharding (before reduction).
             # But here we look at the input to the operation.
             # Actually, for AllReduce, we usually know which axes were partial.
             pass
                 
        return new_kwargs

    def maxpr(
        self,
        shard_values: list[TensorValue],
        mesh: DeviceMesh = None,
        reduce_op: str = "sum",
        reduce_axes: set[str] = None,
    ) -> list[TensorValue]:
        """Reduce across shards (AllReduce)."""
        if not shard_values:
            return []

        if mesh and mesh.is_distributed and len(shard_values) > 1:
            from max.dtype import DType
            from max.graph.ops.allgather import allgather as max_allgather
            from max.graph.type import BufferType
            
            # Signal buffer must be uint8 and large enough (>49KB) to avoid errors
            BUFFER_SIZE = 65536
            signal_buffers = [
                ops.buffer_create(BufferType(DType.uint8, (BUFFER_SIZE,), dev))
                for dev in mesh.device_refs
            ]
            
            # allgather returns list of gathered tensors, one per device
            # Each contains all data concatenated
            gathered = max_allgather(shard_values, signal_buffers, axis=0)
            
            # Now split each gathered tensor back into chunks and reduce
            result_values = []
            chunk_size = shard_values[0].type.shape[0]
            
            for gathered_tensor in gathered:
                # Split the gathered tensor into N chunks
                chunks = []
                for i in range(len(shard_values)):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size
                    chunk = gathered_tensor[start:end]
                    chunks.append(chunk)
                
                # Reduce all chunks locally
                reduced = chunks[0]
                for chunk in chunks[1:]:
                    if reduce_op == "sum": reduced = ops.add(reduced, chunk)
                    elif reduce_op == "max": reduced = ops.max(reduced, chunk)
                    elif reduce_op == "min": reduced = ops.min(reduced, chunk)
                    elif reduce_op == "prod": reduced = ops.mul(reduced, chunk)
                    else: raise ValueError(f"Unknown reduction op: {reduce_op}")
                
                result_values.append(reduced)
            
            return result_values

        if reduce_axes and mesh and not mesh.is_distributed:
             # Use grouped simulation for multi-dim meshes
             return self.simulate_grouped_execution(
                 shard_values, mesh, reduce_axes, reduce_op=reduce_op
             )

        # Simulation mode and single-device fallback (Total reduction)
        result = shard_values[0]
        for sv in shard_values[1:]:
            if reduce_op == "sum": result = ops.add(result, sv)
            elif reduce_op == "max": result = ops.max(result, sv)
            elif reduce_op == "min": result = ops.min(result, sv)
            elif reduce_op == "prod": result = ops.mul(result, sv)
            else: raise ValueError(f"Unknown reduction op: {reduce_op}")

        return [result] * len(shard_values)
        
    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for AllReduce (sum): identity because it's a linear sum across devices."""
        return cotangent

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
            return ShardingSpec(new_dim_specs, partial_sum_axes=set())
            
        new_spec = input_tensor.sharding.clone()
        new_spec.partial_sum_axes.clear()
        
        for ds in new_spec.dim_specs:
             # Remove reduced axes from this dimension's sharding
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
            return self.maxpr(shard_results, mesh=mesh, reduce_op=reduce_op)

        group_axes = [ax for ax in all_axes if ax not in reduce_axes]
        groups = self._group_shards_by_axes(shard_results, mesh, group_axes)

        new_results = [None] * num_shards

        for key, group_members in groups.items():

            group_shards = [val for _, val in group_members]

            if len(group_shards) > 1:
                curr_reduced = self.maxpr(group_shards, mesh=mesh, reduce_op=reduce_op)
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

    def maxpr(
        self,
        shard_values: list[TensorValue],
        mesh: DeviceMesh = None,
        axis_name: str = None,
    ) -> list[TensorValue]:
        """Compute mean across shards."""

        reduced = all_reduce_op.maxpr(shard_values, mesh=mesh)

        if axis_name and mesh:
            axis_size = mesh.get_axis_size(axis_name)
        else:
            axis_size = len(shard_values)

        dtype = reduced[0].type.dtype
        device = reduced[0].type.device
        scale = ops.constant(1.0 / axis_size, dtype, device)
        return [ops.mul(r, scale) for r in reduced]

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
