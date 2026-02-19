# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max.graph import ops

from ..base import (
    AxisOp,
    OpArgs,
    Operation,
    OpKwargs,
    OpResult,
    OpTensorValues,
    ShapeOp,
)
from .axes import unsqueeze

if TYPE_CHECKING:
    from ...core import Tensor
    from ...core.sharding.spec import DeviceMesh, ShardingSpec


def _shape_kwarg_to_local(
    kwargs: OpKwargs, output_sharding: ShardingSpec | None, shard_idx: int
) -> OpKwargs:
    """Convert global 'shape' kwarg to local shape for a given shard."""
    from ...core.sharding.spec import compute_local_shape

    global_shape = kwargs.get("shape")
    if global_shape is None or output_sharding is None:
        return kwargs
    local_shape = compute_local_shape(
        global_shape, output_sharding, device_id=shard_idx
    )
    return {**kwargs, "shape": local_shape}


def _force_replicated_sharding(args: OpArgs, mesh: DeviceMesh | None) -> None:
    """Return (output_spec, input_specs, False) forcing everything replicated."""
    from ...core import Tensor, pytree
    from ...core.sharding.spmd import create_replicated_spec

    leaves = [a for a in pytree.tree_leaves(args) if isinstance(a, Tensor)]
    input_specs = [create_replicated_spec(mesh, len(t.shape)) for t in leaves]
    output_spec = create_replicated_spec(mesh, len(leaves[0].shape))
    return output_spec, input_specs, False


class BroadcastToOp(ShapeOp):
    @property
    def name(self) -> str:
        return "broadcast_to"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        shape = tuple(kwargs["shape"])
        x_rank = len(x.shape)
        if len(shape) < x_rank:
            missing = x_rank - len(shape)
            shape = tuple(x.shape[i] for i in range(missing)) + shape
        return [ops.broadcast_to(x, shape)]

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Broadcast: input dims align to output SUFFIX (numpy semantics)."""
        from ...core.sharding.propagation import OpShardingRuleTemplate

        if not input_shapes:
            return None

        in_shape = input_shapes[0]

        out_shape = kwargs.get("shape")
        if out_shape is None:
            if output_shapes:
                out_shape = output_shapes[0]
            else:
                return None

        in_rank = len(in_shape)
        out_rank = len(out_shape)

        if in_rank == 0:
            out_factors = [f"d{i}" for i in range(out_rank)]
            out_mapping = {i: [out_factors[i]] for i in range(out_rank)}
            in_mapping = {}
            return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
                input_shapes, output_shapes
            )

        out_factors = [f"d{i}" for i in range(out_rank)]
        out_mapping = {i: [out_factors[i]] for i in range(out_rank)}

        in_mapping = {}
        offset = out_rank - in_rank

        for i in range(in_rank):
            out_idx = i + offset
            if out_idx >= 0:
                if in_shape[i] == out_shape[out_idx]:
                    in_mapping[i] = [out_factors[out_idx]]
                else:
                    in_mapping[i] = [f"bcast_{i}"]
            else:
                in_mapping[i] = [f"d_extra_{i}"]

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding, shard_idx: int, args: list
    ) -> dict:
        return _shape_kwarg_to_local(kwargs, output_sharding, shard_idx)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for broadcast_to: sum over broadcasted dimensions."""
        input_shape = tuple(primals[0].shape)
        output_rank = len(outputs[0].shape)
        input_rank = len(input_shape)

        from ...ops.reduction import reduce_sum

        result = cotangents[0]

        # Sum over leading new dimensions
        for _ in range(output_rank - input_rank):
            result = reduce_sum(result, axis=0, keepdims=False)

        # Sum over dimensions that were size 1 and got broadcast
        for i, in_dim in enumerate(input_shape):
            if in_dim == 1:
                result = reduce_sum(result, axis=i, keepdims=True)

        return [result]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        target_shape = tuple(int(d) for d in outputs[0].shape)
        return [broadcast_to(tangents[0], target_shape)]


class ReshapeOp(ShapeOp):
    @property
    def name(self) -> str:
        return "reshape"

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        """Reshape with conservative sharding safety.

        If ANY logical dimension is sharded, we gather the entire tensor.
        """
        from ...core.sharding.spec import DimSpec

        x = args[0]
        shape = kwargs["shape"]

        spec = x.sharding
        if spec:
            batch_dims = x.batch_dims
            logical_rank = len(x.shape)

            has_sharded_logical = False
            for i in range(logical_rank):
                phys_idx = batch_dims + i
                if phys_idx < len(spec.dim_specs) and spec.dim_specs[phys_idx].axes:
                    has_sharded_logical = True
                    break

            if has_sharded_logical:
                new_dim_specs = []
                for i in range(len(spec.dim_specs)):
                    if i < batch_dims:
                        new_dim_specs.append(spec.dim_specs[i].clone())
                    else:
                        new_dim_specs.append(DimSpec([]))

                x = x.with_sharding(spec.mesh, new_dim_specs)

        return super().__call__([x], {"shape": shape})

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        # Nanobind expects a tuple of native ints. The shape received here
        # is the LOCAL physical shape, transformed by _transform_shard_kwargs.
        x = args[0]
        shape = tuple(int(d) for d in kwargs["shape"])
        return [ops.reshape(x, shape)]

    def _resolve_shape(
        self, x_shape: tuple[int, ...], target_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Resolve -1 in target_shape based on input x_shape volume."""
        if -1 not in target_shape:
            return target_shape

        input_size = 1
        for d in x_shape:
            input_size *= int(d)

        resolved = list(target_shape)
        known_size = 1
        neg_idx = -1
        for i, d in enumerate(resolved):
            if d == -1:
                if neg_idx != -1:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_size *= int(d)

        if known_size > 0:
            resolved[neg_idx] = input_size // known_size

        return tuple(resolved)

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding, shard_idx: int, args: list
    ) -> dict:
        """Convert global target shape to local shape for each shard."""
        from ...core import Tensor
        from ...core.sharding.spec import compute_local_shape

        global_shape = kwargs.get("shape")
        if global_shape is None or output_sharding is None:
            return kwargs

        # Resolve -1 if present using the global shape of the input tensor
        if -1 in global_shape and len(args) > 0:
            x = args[0]
            if isinstance(x, Tensor):
                # x.shape returns the global logical shape
                global_shape = self._resolve_shape(x.shape, global_shape)

        local_shape = compute_local_shape(
            global_shape, output_sharding, device_id=shard_idx
        )
        return {**kwargs, "shape": local_shape}

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], Any]:
        """Infer physical shapes for reshape, handling inferred dims (-1)."""
        from ...core.sharding import spec, spmd

        x = args[0]
        # kwargs['shape'] passed here might already include batch dimensions if vmapped.
        target_shape = kwargs.get("shape")

        # 1. Resolve Global Physical Shape (handle -1)
        # We always resolve against the input's physical global shape (which includes batch dims).
        # This handles both standard and VMap cases uniformly.
        resolved_shape = target_shape
        input_shape = None

        if hasattr(x, "physical_global_shape"):
            input_shape = x.physical_global_shape
        elif hasattr(x, "shape"):  # Fallback
            input_shape = x.shape

        if input_shape is not None and -1 in target_shape:
            resolved_shape = self._resolve_shape(input_shape, target_shape)

        # 2. Compute Local Physical Shape for each shard
        # We assume output_sharding matches the rank of resolved_shape.

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        if output_sharding and mesh:
            for i in range(num_shards):
                local = spec.compute_local_shape(
                    resolved_shape, output_sharding, device_id=i
                )
                shapes.append(tuple(int(d) for d in local))
        else:
            shapes = [tuple(int(d) for d in resolved_shape)] * num_shards

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(kwargs.get("shape"))

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Create sharding rule for reshape using greedy factor matching."""
        from ...core.sharding.propagation import OpShardingRuleTemplate

        if not input_shapes:
            return None
        in_shape = input_shapes[0]
        out_shape = kwargs.get("shape")
        if out_shape is None:
            if output_shapes:
                out_shape = output_shapes[0]
            else:
                return None

        in_rank = len(in_shape)
        out_rank = len(out_shape)

        if in_rank >= out_rank:
            factors = [f"d{i}" for i in range(in_rank)]
            in_mapping = {i: [factors[i]] for i in range(in_rank)}
            out_mapping = {}

            factor_idx = 0
            current_prod = 1
            current_factors = []

            for out_dim_idx in range(out_rank):
                target_size = out_shape[out_dim_idx]
                while factor_idx < in_rank:
                    f_size = in_shape[factor_idx]
                    current_factors.append(factors[factor_idx])
                    current_prod *= f_size
                    factor_idx += 1
                    if current_prod == target_size or current_prod > target_size:
                        out_mapping[out_dim_idx] = list(current_factors)
                        current_factors = []
                        current_prod = 1
                        break

            pass
        else:
            factors = [f"d{i}" for i in range(out_rank)]
            out_mapping = {i: [factors[i]] for i in range(out_rank)}
            in_mapping = {}

            factor_idx = 0
            current_prod = 1
            current_factors = []

            for in_dim_idx in range(in_rank):
                target_size = in_shape[in_dim_idx]
                while factor_idx < out_rank:
                    f_size = out_shape[factor_idx]
                    current_factors.append(factors[factor_idx])
                    current_prod *= f_size
                    factor_idx += 1
                    if current_prod == target_size or current_prod > target_size:
                        in_mapping[in_dim_idx] = list(current_factors)
                        current_factors = []
                        current_prod = 1
                        break

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for reshape: reshape cotangent back to input shape."""
        x = primals[0]
        return [reshape(cotangents[0], tuple(x.shape))]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """JVP for reshape: reshape tangent to output shape."""
        target_shape = kwargs.get("shape")
        return [reshape(tangents[0], target_shape)]


class SliceUpdateOp(Operation):
    @property
    def name(self) -> str:
        return "slice_update"

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        x = args[0]
        update = args[1]
        start = kwargs["start"]
        size = kwargs["size"]
        use_buffer_ops = bool(kwargs.get("use_buffer_ops", False))

        # Resolve negative start indices
        shape = x.shape
        _rank = len(shape)
        resolved_start = []
        for i, s in enumerate(start):
            if s < 0:
                s += shape[i]
            resolved_start.append(s)
        resolved_start = tuple(resolved_start)

        batch_dims = x.batch_dims
        slices = [slice(None)] * batch_dims
        for s, sz in zip(resolved_start, size, strict=False):
            end = s + sz
            slices.append(slice(s, end))

        return super().__call__(
            [x, update],
            {
                "slices": tuple(slices),
                "start": resolved_start,
                "size": size,
                "use_buffer_ops": use_buffer_ops,
            },
        )

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        """Slice update kernel.

        - `use_buffer_ops=True`: explicit MAX buffer mutation path.
        - otherwise: functional path (pad + masks).
        """
        x = args[0]
        update = args[1]
        slices = kwargs.get("slices")
        start = kwargs.get("start")
        size = kwargs.get("size")
        use_buffer_ops = bool(kwargs.get("use_buffer_ops", False))

        if use_buffer_ops or start is None or size is None:
            expected_shape: list[int] = []
            for dim, slc in zip(x.shape, slices, strict=False):
                if isinstance(slc, slice):
                    if slc.start is None and slc.stop is None and slc.step is None:
                        expected_shape.append(int(dim))
                    else:
                        slc_start = 0 if slc.start is None else int(slc.start)
                        slc_stop = int(dim) if slc.stop is None else int(slc.stop)
                        expected_shape.append(slc_stop - slc_start)
                else:
                    expected_shape.append(1)

            if tuple(update.shape) != tuple(expected_shape):
                update = ops.broadcast_to(update, tuple(expected_shape))

            x_buffer = ops.buffer_create(x.type.as_buffer())
            ops.buffer_store(x_buffer, x)
            ops.buffer_store_slice(x_buffer, update, slices)
            return [ops.buffer_load(x_buffer)]

        # Functional implementation
        rank = len(x.shape)
        update_rank = len(update.shape)
        prefix_rank = max(0, update_rank - rank)
        if start is not None and size is not None and len(start) < rank:
            pad = rank - len(start)
            start = (0,) * pad + tuple(start)
            size = [int(x.shape[i]) for i in range(pad)] + list(size)
        paddings = []

        # We need update shape for ones_like
        update_shape = []

        for i in range(prefix_rank):
            paddings.extend([0, 0])
            update_shape.append(int(update.shape[i]))

        for i in range(rank):
            s = start[i]
            sz = size[i]
            # Handle potential negative start (though __call__ resolves it)
            # x.shape[i] might be a Dim object. We assume it converts to int validly here if needed,
            # or ops.pad handles it.
            # But ops.pad expects `int` usually.
            # Let's try to access `.value` if it's a constant Dim, or cast to int.
            dim_len = int(x.shape[i])

            before = s
            after = dim_len - (s + sz)
            paddings.extend([before, after])

            update_shape.append(int(sz))

        if tuple(update.shape) != tuple(update_shape):
            update = ops.broadcast_to(update, tuple(update_shape))

        padded_update = ops.pad(update, paddings, value=0.0)
        scalar_one = ops.constant(1.0, dtype=x.dtype, device=x.device)
        ones_update = ops.broadcast_to(scalar_one, tuple(update_shape))
        padded_mask = ops.pad(ones_update, paddings, value=0.0)

        inv_mask = ops.sub(
            ops.constant(1.0, dtype=x.dtype, device=x.device), padded_mask
        )
        masked_x = ops.mul(x, inv_mask)
        result = ops.add(masked_x, padded_update)

        return [result]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for slice_update (same as input x)."""
        from ...core.sharding import spmd

        x = args[0]
        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            idx = i if i < x.num_shards else 0
            s = x.physical_local_shape(idx)
            if s is not None:
                shapes.append(tuple(int(d) for d in s))
            else:
                raise RuntimeError(
                    f"Could not determine physical shape for input x in {self.name}"
                )

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])

    def infer_sharding_spec(
        self,
        args: list,
        mesh: Any,
        kwargs: dict = None,
    ) -> tuple[Any | None, list[Any | None], bool]:
        """Explicitly force everything to be Replicated."""
        return _force_replicated_sharding(args, mesh)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        # primals = [x, update]
        _x_in = primals[0]
        u_in = primals[1]
        start = kwargs.get("start")
        size = kwargs.get("size")

        from ..creation import zeros_like
        from .shape import slice_tensor, slice_update

        # Compute dx: cotangent with zeros at the update region
        u_zeros = zeros_like(u_in)
        dx = slice_update(cotangents[0], u_zeros, start=start, size=size)

        # Compute du: slice of cotangent corresponding to the update
        du = slice_tensor(cotangents[0], start=start, size=size)

        return [dx, du, None, None]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        start = kwargs.get("start")
        size = kwargs.get("size")
        tx, t_update = tangents[0], tangents[1]

        from .shape import slice_update

        return [slice_update(tx, t_update, start=start, size=size)]


class SliceTensorOp(Operation):
    @property
    def name(self) -> str:
        return "slice_tensor"

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        x = args[0]
        start = kwargs["start"]
        size = kwargs["size"]

        # Resolve negative start indices
        shape = x.shape
        _rank = len(shape)
        resolved_start = []
        for i, s in enumerate(start):
            if s < 0:
                s += shape[i]
            resolved_start.append(s)
        resolved_start = tuple(resolved_start)

        batch_dims = x.batch_dims
        slices = [slice(None)] * batch_dims
        for s, sz in zip(resolved_start, size, strict=False):
            end = s + sz
            slices.append(slice(s, end))

        return super().__call__(
            [x], {"slices": tuple(slices), "start": resolved_start, "size": size}
        )

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        slices = kwargs["slices"]
        return [ops.slice_tensor(x, slices)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for slice_tensor."""
        from ...core.sharding import spmd

        x = args[0]
        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        # Get batch prefix from input's physical shape
        batch_dims = x.batch_dims if hasattr(x, "batch_dims") else 0
        batch_prefix = ()
        if batch_dims > 0:
            phys = x.physical_local_shape(0)
            if phys is not None:
                batch_prefix = tuple(int(d) for d in phys[:batch_dims])

        shapes = []
        for i in range(num_shards):
            # SliceTensorOp uses _transform_shard_kwargs to determine local slice
            local_kwargs = self._transform_shard_kwargs(
                kwargs, output_sharding, i, args
            )
            local_size = local_kwargs.get("size")
            if local_size is not None:
                logical_shape = tuple(int(d) for d in local_size)
                shapes.append(batch_prefix + logical_shape)
            else:
                raise RuntimeError(
                    f"Could not determine local physical shape for {self.name}"
                )

        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])

    def infer_sharding_spec(
        self,
        args: list,
        mesh: Any,
        kwargs: dict = None,
    ) -> tuple[Any | None, list[Any | None], bool]:
        """Propagate input sharding spec."""
        from ...core import Tensor

        if not args:
            return None, [], False

        x = args[0]
        if isinstance(x, Tensor) and x.sharding:
            return x.sharding, [x.sharding], False

        return None, [None], False

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        x = primals[0]

        start = kwargs.get("start")
        size = kwargs.get("size")

        from ..creation import zeros_like
        from .shape import slice_update

        z = zeros_like(x)
        grad = slice_update(z, cotangents[0], start=start, size=size)

        return [grad, None, None]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        start = kwargs.get("start")
        size = kwargs.get("size")
        from .shape import slice_tensor

        return [slice_tensor(tangents[0], start=start, size=size)]

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding: Any, shard_idx: int, args: list
    ) -> dict:
        """Transform slice args to be local to the shard."""
        import math

        # 1. Get Global Slice Request
        global_start = kwargs.get("start")
        global_size = kwargs.get("size")
        if global_start is None or global_size is None:
            return kwargs

        # 2. Get Input Info
        if not args:
            return kwargs
        x = args[0]
        # We need the underlying sharding info from the input tensor
        if not hasattr(x, "sharding") or x.sharding is None:
            return kwargs

        input_sharding = x.sharding
        # Use logical shape (Tensor.shape) which corresponds to global logical shape
        input_global_shape = x.shape

        # 3. Compute Input Shard Interval for each dim
        mesh = input_sharding.mesh
        # Map logical shard index to device ID in the mesh
        # mesh.devices is a list of device IDs corresponding to the flattened mesh
        device_id = (
            mesh.devices[shard_idx] if shard_idx < len(mesh.devices) else shard_idx
        )

        local_start_indices = []
        local_sizes = []

        batch_dims = getattr(x, "batch_dims", 0)

        for dim_idx, (g_start, g_size) in enumerate(
            zip(global_start, global_size, strict=False)
        ):
            phys_idx = batch_dims + dim_idx
            dim_spec = (
                input_sharding.dim_specs[phys_idx]
                if phys_idx < len(input_sharding.dim_specs)
                else None
            )
            global_len = int(input_global_shape[dim_idx])

            # --- Determine Shard Interval ---
            if not dim_spec or not dim_spec.axes:
                shard_start = 0
                shard_end = global_len
            else:
                total_shards = 1
                my_shard_index = 0
                for axis_name in dim_spec.axes:
                    size = mesh.get_axis_size(axis_name)
                    coord = mesh.get_coordinate(device_id, axis_name)
                    my_shard_index = (my_shard_index * size) + coord
                    total_shards *= size

                chunk_size = math.ceil(global_len / total_shards)
                shard_start = my_shard_index * chunk_size
                shard_end = min(shard_start + chunk_size, global_len)

            # --- Intersect with Global Slice ---
            req_start = g_start
            req_end = g_start + g_size

            # Intersection in Global Coords
            inter_start = max(shard_start, req_start)
            inter_end = min(shard_end, req_end)

            if inter_end <= inter_start:
                local_len = 0
                local_off = 0
            else:
                local_len = inter_end - inter_start
                # Map to Local Coords (relative to Shard Start)
                local_off = inter_start - shard_start

            local_start_indices.append(local_off)
            local_sizes.append(local_len)

        # 4. Update kwargs
        new_kwargs = kwargs.copy()
        new_kwargs["start"] = tuple(local_start_indices)
        new_kwargs["size"] = tuple(local_sizes)

        # Reconstruct slices for kernel (assuming full rank coverage)
        final_slices = []

        # Prepend batch dims if present
        if hasattr(x, "batch_dims"):
            final_slices.extend([slice(None)] * x.batch_dims)

        for s, sz in zip(local_start_indices, local_sizes, strict=False):
            final_slices.append(slice(s, s + sz))

        new_kwargs["slices"] = tuple(final_slices)

        return new_kwargs


class ConcatenateOp(AxisOp):
    """Concatenate tensors along an axis."""

    @property
    def name(self) -> str:
        return "concatenate"

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        if not args:
            raise ValueError("concatenate expects at least one tensor")

        return super().__call__(args, kwargs)

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        axis = kwargs.get("axis", 0)
        return [ops.concat(args, axis=axis)]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        """Infer physical shapes for concatenate."""
        from ...core.sharding import spmd

        tensors = args
        axis = kwargs.get("axis", 0)

        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1

        shapes = []
        for i in range(num_shards):
            total_axis_size = 0
            ref_shape = None
            for t in tensors:
                idx = i if i < t.num_shards else 0
                s = t.physical_local_shape(idx)
                if s is not None:
                    norm_axis = axis if axis >= 0 else len(s) + axis
                    total_axis_size += int(s[norm_axis])
                    if ref_shape is None:
                        ref_shape = [int(d) for d in s]
                else:
                    raise RuntimeError(
                        f"Could not determine physical shape for input in {self.name}"
                    )

            if ref_shape is not None:
                ref_shape[norm_axis] = total_axis_size
                shapes.append(tuple(ref_shape))

        dtypes, devices = self._build_shard_metadata(args[0], mesh, num_shards)
        return shapes, dtypes, devices

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Concatenate: inputs share factors on non-concat axis.

        Concat axis uses a SHARED factor 'c_concat' across all inputs and output.
        """
        from ...core.sharding.propagation import OpShardingRuleTemplate

        if not input_shapes:
            return None

        rank = len(input_shapes[0])
        num_inputs = len(input_shapes)
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis += rank

        input_mappings = []
        for _input_idx in range(num_inputs):
            mapping = {}
            for dim in range(rank):
                if dim == axis:
                    mapping[dim] = ["c_concat"]
                else:
                    mapping[dim] = [f"d{dim}"]
            input_mappings.append(mapping)

        out_mapping = {}
        for dim in range(rank):
            if dim == axis:
                out_mapping[dim] = ["c_concat"]
            else:
                out_mapping[dim] = [f"d{dim}"]

        return OpShardingRuleTemplate(input_mappings, [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for concatenate: split cotangent along axis."""
        from ...ops.view import slice_tensor

        axis = kwargs.get("axis", 0)

        # primals is a flat list of input tensors [t1, t2, ...]
        # Calculate split indices based on primal shapes
        start = 0
        cotangent_slices = []

        # Handle negative axis
        if axis < 0:
            axis += len(primals[0].shape)

        for x in primals:
            # We slice along 'axis'
            dim_size = int(x.shape[axis])

            # Construct start/size for slice_tensor
            # slice_tensor takes start=[...], size=[...]
            rank = len(x.shape)

            starts = [0] * rank
            starts[axis] = start

            sizes = [int(d) for d in cotangents[0].shape]  # Use full shape
            sizes[axis] = dim_size  # Update split axis size

            slc = slice_tensor(cotangents[0], start=starts, size=sizes)
            cotangent_slices.append(slc)

            start += dim_size

        # Return flat list of gradient tensors, one per primal
        return cotangent_slices

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        axis = kwargs.get("axis", 0)
        # tangents is a flat list of tangent tensors
        return [concatenate(tangents, axis=axis)]


_broadcast_to_op = BroadcastToOp()
_reshape_op = ReshapeOp()
_slice_tensor_op = SliceTensorOp()
_concatenate_op = ConcatenateOp()


def broadcast_to(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    in_logical_rank = len(x.shape)
    out_logical_rank = len(shape)

    if in_logical_rank < out_logical_rank:
        for _ in range(out_logical_rank - in_logical_rank):
            x = unsqueeze(x, axis=0)

    return _broadcast_to_op([x], {"shape": shape})[0]


def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _reshape_op([x], {"shape": shape})[0]


_slice_update_op = SliceUpdateOp()


def slice_tensor(x: Tensor, start: Any, size: Any) -> Tensor:
    return _slice_tensor_op([x], {"start": start, "size": size})[0]


def slice_update(x: Tensor, update: Tensor, start: Any, size: Any) -> Tensor:
    """Update a slice of x with new values."""
    from ..base import ensure_tensor

    x = ensure_tensor(x)
    update = ensure_tensor(update)
    return _slice_update_op(
        [x, update],
        {"start": start, "size": size, "use_buffer_ops": False},
    )[0]


def slice_update_inplace(x: Tensor, update: Tensor, start: Any, size: Any) -> Tensor:
    """Slice update lowered through explicit MAX buffer mutation ops."""
    from ..base import ensure_tensor

    x = ensure_tensor(x)
    update = ensure_tensor(update)
    return _slice_update_op(
        [x, update],
        {"start": start, "size": size, "use_buffer_ops": True},
    )[0]


def concatenate(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    return _concatenate_op(list(tensors), {"axis": axis})[0]


def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along a new axis."""
    expanded = [unsqueeze(t, axis=axis) for t in tensors]
    return concatenate(expanded, axis=axis)


class BroadcastToPhysicalOp(ShapeOp):
    @property
    def name(self) -> str:
        return "broadcast_to_physical"

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        shape = tuple(kwargs["shape"])
        x_rank = len(x.shape)
        if len(shape) < x_rank:
            missing = x_rank - len(shape)
            shape = tuple(x.shape[i] for i in range(missing)) + shape
        return [ops.broadcast_to(x, shape)]

    def __call__(self, args: OpArgs, kwargs: OpKwargs) -> OpResult:
        """Broadcast physical shape, auto-incrementing batch_dims when rank increases.

        Args:
            args: [x] - Input tensor
            kwargs: {"shape": target_shape} - Target GLOBAL physical shape
        """
        from .axes import unsqueeze_physical

        x = args[0]
        shape = kwargs["shape"]

        in_phys = x.physical_global_shape or x.local_shape
        in_shape = tuple(int(d) for d in (in_phys if in_phys is not None else x.shape))
        target_shape = tuple(int(d) for d in shape)

        if os.environ.get("NABLA_DEBUG_PHYS", "0") == "1":
            print(f"[NABLA_DEBUG_PHYS] broadcast_to_physical.call: x.shape={tuple(int(d) for d in x.shape)} x.batch_dims={x.batch_dims} in_phys={in_shape} target_phys={target_shape}")

        # Note: target_shape here is LOGICAL for the current call level.
        # adapt_kwargs (via ShapeOp) will prepend existing batch_dims.
        
        in_logical_rank = len(in_shape) - int(x.batch_dims)
        out_logical_rank = len(target_shape)
        added_dims = max(0, out_logical_rank - in_logical_rank)

        if added_dims > 0:
            # We insert dimensions at the front of logical.
            # Since unsqueeze_physical is an AxisOp, passing axis=0 will be 
            # shifted by adapt_kwargs to target the correct physical index (x.batch_dims).
            for _ in range(added_dims):
                x = unsqueeze_physical(x, axis=0)

        results = super().__call__([x], {"shape": target_shape})
        return results

    def adapt_kwargs(self, args: OpArgs, kwargs: OpKwargs, batch_dims: int) -> OpKwargs:
        # BroadcastToPhysicalOp takes a physical shape. We don't want ShapeOp 
        # to automatically prepend batch_dims because they might already be there.
        # Instead, we depend on the caller or JVP rule to provide the full physical shape.
        return kwargs

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        shape = kwargs.get("shape")
        if shape is None:
             phys = x.physical_global_shape or x.local_shape
             shape = tuple(int(d) for d in phys)
        
        from ...core.sharding import spmd
        mesh = spmd.get_mesh_from_args(args)
        num_shards = len(mesh.devices) if mesh else 1
        
        # Currently we assume the provided shape IS the global physical shape.
        # Replicated fallback:
        shapes = [tuple(int(d) for d in shape)] * num_shards
        dtypes, devices = self._build_shard_metadata(x, mesh, num_shards)
        return shapes, dtypes, devices

    def infer_output_rank(
        self, input_shapes: tuple[tuple[int, ...], ...], **kwargs
    ) -> int:
        """Output rank is the target shape's rank."""
        target_shape = kwargs.get("shape")
        if target_shape is not None:
            return len(target_shape)

        return len(input_shapes[0]) if input_shapes else 0

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ...core.sharding.propagation import OpShardingRuleTemplate

        in_shape = input_shapes[0]

        if output_shapes is not None:
            out_shape = output_shapes[0]
        else:
            out_shape = kwargs.get("shape")

        if out_shape is None:
            raise ValueError(
                "BroadcastToPhysicalOp requires 'shape' kwarg or output_shapes"
            )

        in_rank = len(in_shape)
        out_rank = len(out_shape)
        offset = out_rank - in_rank

        factors = [f"d{i}" for i in range(out_rank)]
        out_mapping = {i: [factors[i]] for i in range(out_rank)}
        in_mapping = {}

        for i in range(in_rank):
            out_dim_idx = i + offset
            in_dim_size = in_shape[i]
            out_dim_size = out_shape[out_dim_idx]

            if in_dim_size == 1 and out_dim_size > 1:
                in_mapping[i] = []
            else:
                in_mapping[i] = [factors[out_dim_idx]]

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        """VJP for broadcast_to_physical: sum over broadcasted dimensions."""
        from ...ops.reduction import reduce_sum_physical

        primal_phys = tuple(
            int(d) for d in (primals[0].physical_global_shape or primals[0].local_shape)
        )
        output_phys = tuple(
            int(d) for d in (outputs[0].physical_global_shape or outputs[0].local_shape)
        )

        in_rank = len(primal_phys)
        out_rank = len(output_phys)
        offset = out_rank - in_rank

        result = cotangents[0]
        # 1. Sum over leading new dimensions (which became batch dims)
        for _ in range(offset):
            result = reduce_sum_physical(result, axis=0, keepdims=False)

        # 2. Sum over dimensions that were size 1 and got broadcast physically
        for i in reversed(range(in_rank)):
            if primal_phys[i] == 1 and output_phys[i + offset] > 1:
                result = reduce_sum_physical(result, axis=i, keepdims=True)

        return [result]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        tx = tangents[0]
        target_shape = outputs[0].physical_global_shape or outputs[0].local_shape
        target_shape = tuple(int(d) for d in target_shape)
        return [broadcast_to_physical(tx, target_shape)]

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding, shard_idx: int, args: list
    ) -> dict:
        return _shape_kwarg_to_local(kwargs, output_sharding, shard_idx)


class RebindOp(Operation):
    """Rebind a tensor to a new symbolic shape/layout."""

    @property
    def name(self) -> str:
        return "rebind"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        # rebind doesn't change physical data, just metadata
        x = args[0]
        return (
            [tuple(x.physical_local_shape(i)) for i in range(x.num_shards)],
            [x.dtype] * x.num_shards,
            [x.device] * x.num_shards,
        )

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        shape = kwargs["shape"]
        extra = {k: v for k, v in kwargs.items() if k != "shape"}
        return [ops.rebind(x, shape, **extra)]

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        return [cotangents[0]]


class PadOp(Operation):
    """Pad a tensor with a constant value."""

    @property
    def name(self) -> str:
        return "pad"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        paddings = kwargs.get("paddings")  # List of (before, after) for LOGICAL dims

        # Get batch prefix from input's physical shape
        batch_dims = x.batch_dims if hasattr(x, "batch_dims") else 0

        shapes = []
        for i in range(x.num_shards):
            in_local = x.physical_local_shape(i)

            # Start with batch dims (unchanged by padding)
            out_local = (
                [int(in_local[d]) for d in range(batch_dims)]
                if in_local is not None
                else []
            )

            # Apply paddings to logical dims only
            for d, (before, after) in enumerate(paddings):
                phys_d = batch_dims + d
                sz = int(in_local[phys_d]) if in_local is not None else 0
                out_local.append(sz + int(before) + int(after))
            shapes.append(tuple(out_local))

        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def kernel(self, args: OpTensorValues, kwargs: OpKwargs) -> OpTensorValues:
        x = args[0]
        paddings = kwargs.get("paddings")
        mode = kwargs.get("mode", "constant")
        value = kwargs.get("value", 0.0)
        flat_paddings = []

        # Handle vmap rank mismatch
        # If input has higher rank than paddings imply, prepend (0, 0) for batch dims.
        # paddings list has N elements for N dimensions.
        input_rank = len(x.shape)
        paddings_rank = len(paddings)
        if input_rank > paddings_rank:
            extra_dims = input_rank - paddings_rank
            for _ in range(extra_dims):
                flat_paddings.extend([0, 0])

        for p in paddings:
            flat_paddings.extend([int(p[0]), int(p[1])])

        return [ops.pad(x, flat_paddings, mode=mode, value=value)]

    def infer_sharding_spec(
        self,
        args: list,
        mesh: Any,
        kwargs: dict = None,
    ) -> tuple[Any | None, list[Any | None], bool]:
        """Force replicated: pad doesn't support sharded dims that are being padded."""
        return _force_replicated_sharding(args, mesh)

    def vjp_rule(
        self, primals: OpArgs, cotangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        paddings = kwargs.get("paddings")
        x = primals[0]

        start = [p[0] for p in paddings]
        size = [int(d) for d in x.shape]

        from .shape import slice_tensor

        return [slice_tensor(cotangents[0], start=start, size=size)]

    def jvp_rule(
        self, primals: OpArgs, tangents: OpArgs, outputs: OpArgs, kwargs: OpKwargs
    ) -> OpResult:
        paddings = kwargs.get("paddings")
        mode = kwargs.get("mode", "constant")
        return [pad(tangents[0], paddings=paddings, mode=mode, value=0.0)]


_rebind_op = RebindOp()
_pad_op = PadOp()
_broadcast_to_physical_op = BroadcastToPhysicalOp()


def broadcast_to_physical(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _broadcast_to_physical_op([x], {"shape": shape})[0]


def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten a range of dimensions into a single dimension using reshape."""
    shape = x.shape
    rank = len(shape)
    if start_dim < 0:
        start_dim += rank
    if end_dim < 0:
        end_dim += rank
    if start_dim >= end_dim:
        return x

    new_shape = list(shape[:start_dim])
    flat_dim = 1
    for i in range(start_dim, end_dim + 1):
        # Cast to int to resolve any lazy Dim objects before arithmetic
        flat_dim *= int(shape[i])
    new_shape.append(flat_dim)
    new_shape.extend(shape[end_dim + 1 :])

    return reshape(x, tuple(new_shape))


def rebind(x: Tensor, shape: tuple[int, ...], **kwargs) -> Tensor:
    return _rebind_op([x], {"shape": shape, **kwargs})[0]


def pad(
    x: Tensor,
    paddings: list[tuple[int, int]] = None,
    mode: str = "constant",
    value: float = 0.0,
    **kwargs,
) -> Tensor:
    paddings = paddings if paddings is not None else kwargs.get("pad_width")
    if paddings is None:
        raise ValueError("pad() requires paddings or pad_width")
    return _pad_op([x], {"paddings": paddings, "mode": mode, "value": value})[0]


__all__ = [
    "broadcast_to",
    "reshape",
    "slice_tensor",
    "concatenate",
    "stack",
    "BroadcastToPhysicalOp",
    "broadcast_to_physical",
    "SliceTensorOp",
    "flatten",
    "rebind",
    "pad",
]
