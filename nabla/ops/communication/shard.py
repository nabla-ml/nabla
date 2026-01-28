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

        if not x.sharding:
            # If input was not sharded (replicated), we gather/replicate the cotangent
            from .all_gather import gather_all_axes

            return gather_all_axes(cotangent)

        # Use the smart shard function (defined below) to handle transition.
        # Note: reshard name is deprecated, we use universal shard() now.
        return shard(
            cotangent,
            x.sharding.mesh,
            x.sharding.dim_specs,
            replicated_axes=x.sharding.replicated_axes,
        )

    def infer_sharding_spec(self, args, mesh, kwargs):
        spec = kwargs.get("spec")
        if spec is None:
            # Fallback for new flow where dim_specs is passed directly
            if "dim_specs" in kwargs:
                from ...core.sharding.spec import ShardingSpec
                spec = ShardingSpec(mesh, kwargs["dim_specs"], 
                                  replicated_axes=kwargs.get("replicated_axes", set()))
            else:
                # Should we raise? Or return defaults?
                pass
                
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

    def physical_execute(self, args: tuple, kwargs: dict) -> Any:
        """Physical execution for ShardOp.
        
        Derives all physical metadata from args/kwargs and slices the input.
        Returns raw shard values as a tuple: (values, output_spec, mesh).
        """
        from ...core.sharding.spec import ShardingSpec, needs_reshard, compute_global_shape
        from ...core.sharding import spmd
        from ...core import Tensor
        from ...core import GRAPH
        from max import graph as g

        with GRAPH.graph:
            x = args[0]
            
            # Handle positional arguments for mesh/dim_specs
            if len(args) > 1:
                mesh = args[1]
            else:
                mesh = kwargs.get("mesh")
                
            if len(args) > 2:
                dim_specs = args[2]
            else:
                dim_specs = kwargs.get("dim_specs")
            
            replicated_axes = kwargs.get("replicated_axes") or set()
            
            # We need to construct the target spec to check idempotency and return it
            target_spec = ShardingSpec(
                mesh, dim_specs, replicated_axes=replicated_axes
            )
            # 1. Idempotency Check
            if isinstance(x, Tensor) and x.sharding:
                if not needs_reshard(x.sharding, target_spec):
                    # Identity OP: Just return values.
                    return (x.values, target_spec, mesh)
            
            # 2. Global Shape Determination
            global_shape = None
            if isinstance(x, Tensor):
                local = x.physical_local_shape(0)
                if local is not None and x.sharding:
                    global_shape = compute_global_shape(tuple(local), x.sharding)
                elif local is not None:
                    global_shape = tuple(int(d) for d in local)
                # Fallback if uninitialized?
                if global_shape is None:
                    global_shape = tuple(int(d) for d in x.shape) # Logical fallback
            
            if global_shape is None and "global_shape" in kwargs:
                 global_shape = kwargs["global_shape"]

            # 3. Kernel Execution (Slicing)
            x_input = x
            if isinstance(x, Tensor):
                 vals = x.values
                 if vals:
                     x_input = vals[0]
                 else:
                     raise ValueError("ShardOp input tensor missing values.")
            elif not isinstance(x, g.TensorValue):
                 x_input = g.TensorValue(x)
            
            # maxpr expects a single value input usually (replicated)
            shard_values = self.maxpr(
                x_input, mesh, dim_specs, global_shape=global_shape, **kwargs
            )
             
            return (shard_values, target_spec, mesh)

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
            # No hydrate needed, checked by caller or covered by graph context
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


shard_op = ShardOp()



def create_replicated_spec(mesh: DeviceMesh, rank: int) -> ShardingSpec:
    """Create a fully replicated sharding spec."""
    from ...core.sharding.spec import DimSpec, ShardingSpec

    return ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])


def shard(
    x,
    mesh: DeviceMesh,
    dim_specs: list[DimSpec],
    replicated_axes: set[str] | None = None,
    **kwargs,
):
    """Shard a tensor according to the given mesh and dimension specs.
    
    This operation is "smart":
    1. If the input is already sharded differently, it inserts necessary
       communication (AllGather, AllReduce) to transition to the valid state.
    2. Then it applies the physical slicing (ShardOp) to reach the target distribution.
    """
    from ...core import Tensor
    from ...core.sharding.spec import ShardingSpec, needs_reshard, DimSpec

    
    # UI Convenience: Handle implicit batch dimensions in spec
    if isinstance(x, Tensor):
        batch_dims = x.batch_dims
        current_rank = len(x.shape)
        if batch_dims > 0:
            if len(dim_specs) == current_rank:
                batch_specs = [DimSpec([], is_open=True) for _ in range(batch_dims)]
                if x.sharding:
                    current_s = x.sharding
                    if len(current_s.dim_specs) >= batch_dims:
                        for i in range(batch_dims):
                            batch_specs[i] = current_s.dim_specs[i].clone()
                dim_specs = batch_specs + list(dim_specs)

    target_spec = ShardingSpec(
        mesh, dim_specs, replicated_axes=replicated_axes or set()
    )

    if not isinstance(x, Tensor) or not x.sharding:
         # If not a tensor or not sharded, treat as fresh slicing (legacy behavior)
         return shard_op(x, mesh, dim_specs, replicated_axes=replicated_axes, **kwargs)

    # === Transition Logic (merged from ReshardOp) ===
    from_spec = x.sharding
    to_spec = target_spec

    if not needs_reshard(from_spec, to_spec):
         return shard_op(x, mesh, dim_specs, replicated_axes=replicated_axes, **kwargs)

    result = x

    # 1. Expansion (AllGather)
    for dim in range(len(from_spec.dim_specs)):
        from_axes = (
            set(from_spec.dim_specs[dim].axes)
            if dim < len(from_spec.dim_specs)
            else set()
        )
        to_axes = (
            set(to_spec.dim_specs[dim].axes) if dim < len(to_spec.dim_specs) else set()
        )
        axes_to_remove = from_axes - to_axes

        # Handle partial sums
        if from_spec.dim_specs[dim].partial:
            from .all_reduce import all_reduce
            target_is_partial = (
                dim < len(to_spec.dim_specs) and to_spec.dim_specs[dim].partial
            )
            if not target_is_partial:
                result = all_reduce(result)
                continue

        if axes_to_remove:
            from .all_gather import all_gather
            result = all_gather(result, axis=dim)

    # 2. Ghost Axes Reduction
    for ghost_ax in from_spec.partial_sum_axes:
        if ghost_ax not in to_spec.partial_sum_axes:
            from .all_reduce import all_reduce
            result = all_reduce(result)

    # 3. Contraction (ShardOp)
    # We pass _bypass_idempotency=True to force the shard op even if specs look similar
    # (though usually the shape change prevents that confusion, explicit is better)
    return shard_op(
        result,
        mesh,
        dim_specs,
        replicated_axes=replicated_axes,
        **kwargs
    )
