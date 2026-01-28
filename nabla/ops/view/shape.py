# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from ..base import LogicalAxisOperation, LogicalShapeOperation, Operation
from .axes import unsqueeze

if TYPE_CHECKING:
    from ...core import Tensor


class BroadcastToOp(LogicalShapeOperation):
    @property
    def name(self) -> str:
        return "broadcast_to"

    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:

        return ops.broadcast_to(x, shape)

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
        self, kwargs: dict, output_sharding, shard_idx: int, args: tuple
    ) -> dict:
        """Convert global target shape to local shape for each shard."""
        from ...core.sharding.spec import compute_local_shape

        global_shape = kwargs.get("shape")
        if global_shape is None or output_sharding is None:
            return kwargs

        local_shape = compute_local_shape(
            global_shape, output_sharding, device_id=shard_idx
        )
        return {**kwargs, "shape": local_shape}

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for broadcast_to: sum over broadcasted dimensions."""
        input_shape = tuple(primals.shape)
        output_rank = len(output.shape)
        input_rank = len(input_shape)

        from ...ops.reduction import reduce_sum

        result = cotangent

        # Sum over leading new dimensions
        for _ in range(output_rank - input_rank):
            result = reduce_sum(result, axis=0, keepdims=False)

        # Sum over dimensions that were size 1 and got broadcast
        for i, in_dim in enumerate(input_shape):
            if in_dim == 1:
                result = reduce_sum(result, axis=i, keepdims=True)

        return result

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for broadcast_to: broadcast the tangent."""
        target_shape = output.op_kwargs.get("shape")
        return broadcast_to(tangents, target_shape)


class ReshapeOp(LogicalShapeOperation):
    @property
    def name(self) -> str:
        return "reshape"

    def __call__(self, x: Tensor, *, shape: tuple[int, ...]) -> Tensor:
        """Reshape with conservative sharding safety.

        If ANY logical dimension is sharded, we gather the entire tensor.
        """
        from ...core.sharding.spec import DimSpec

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

        return super().__call__(x, shape=shape)

    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.reshape(x, shape)

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

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding, shard_idx: int, args: tuple
    ) -> dict:
        """Convert global target shape to local shape for each shard."""
        from ...core.sharding.spec import compute_local_shape

        global_shape = kwargs.get("shape")
        if global_shape is None or output_sharding is None:
            return kwargs

        local_shape = compute_local_shape(
            global_shape, output_sharding, device_id=shard_idx
        )
        return {**kwargs, "shape": local_shape}

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for reshape: reshape cotangent back to input shape."""
        x = primals
        return reshape(cotangent, tuple(x.shape))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for reshape: reshape tangent to output shape."""
        target_shape = output.op_kwargs.get("shape")
        return reshape(tangents, target_shape)



class SliceUpdateOp(Operation):
    @property
    def name(self) -> str:
        return "slice_update"

    def __call__(self, x: Tensor, update: Tensor, *, start: Any, size: Any) -> Tensor:
        # Resolve negative start indices
        shape = x.shape
        rank = len(shape)
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
            x, update, slices=tuple(slices), start=resolved_start, size=size
        )

    def maxpr(
        self,
        x: TensorValue,
        update: TensorValue,
        *,
        slices: tuple[slice, ...],
        **kwargs,
    ) -> TensorValue:
        """Update using pre-computed slices."""
        x_buffer = ops.buffer_create(x.type.as_buffer())
        ops.buffer_store(x_buffer, x)
        ops.buffer_store_slice(x_buffer, update, slices)
        return ops.buffer_load(x_buffer)

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])
        
    def infer_sharding_spec(
        self,
        args: tuple,
        mesh: Any,
        kwargs: dict = None,
    ) -> tuple[Any | None, list[Any | None], bool]:
        """Explicitly force everything to be Replicated."""
        from ...core.sharding.spmd import create_replicated_spec
        from ...core import Tensor, pytree

        leaves = [a for a in pytree.tree_leaves(args) if isinstance(a, Tensor)]
        input_specs = []
        for t in leaves:
            rank = len(t.shape)
            input_specs.append(create_replicated_spec(mesh, rank))

        # Output is also replicated
        output_rank = len(leaves[0].shape)
        output_spec = create_replicated_spec(mesh, output_rank)

        return output_spec, input_specs, False

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        # primals = (x, update)
        x_in = primals[0]
        u_in = primals[1]
        start = output.op_kwargs.get("start")
        size = output.op_kwargs.get("size")

        from ..creation import zeros_like
        from .shape import slice_tensor, slice_update

        # Compute dx: cotangent with zeros at the update region
        u_zeros = zeros_like(u_in)
        dx = slice_update(cotangent, u_zeros, start=start, size=size)

        # Compute du: slice of cotangent corresponding to the update
        du = slice_tensor(cotangent, start=start, size=size)

        return (dx, du, None, None)


class SliceTensorOp(Operation):
    @property
    def name(self) -> str:
        return "slice_tensor"

    def __call__(self, x: Tensor, *, start: Any, size: Any) -> Tensor:
        # Resolve negative start indices
        shape = x.shape
        rank = len(shape)
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

        return super().__call__(x, slices=tuple(slices), start=resolved_start, size=size)

    def maxpr(
        self, x: TensorValue, *, slices: tuple[slice, ...], **kwargs
    ) -> TensorValue:
        return ops.slice_tensor(x, slices)

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])

    def infer_sharding_spec(
        self,
        args: tuple,
        mesh: Any,
        kwargs: dict = None,
    ) -> tuple[Any | None, list[Any | None], bool]:
        """Propagate input sharding spec."""
        from ...core import Tensor
        
        if not args:
            return None, [], False
            
        x = args[0]
        if isinstance(x, Tensor) and x.sharding:
            # Slice preserves rank and dimension alignment (mostly), 
            # so we can propagate the spec.
            # NOTE: If we slice a sharded dimension to size 1 (or small),
            # it technically remains sharded (distributed scalar/small tensor).
            return x.sharding, [x.sharding], False
            
        return None, [None], False

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        # VJP: slice_tensor(x, start, size) -> slice_update(zeros_like(x), cotangent, start, size)
        if isinstance(primals, (list, tuple)):
            x = primals[0]
        else:
            x = primals

        start = output.op_kwargs.get("start")
        size = output.op_kwargs.get("size")

        from ..creation import zeros_like
        from .shape import slice_update

        z = zeros_like(x)
        grad = slice_update(z, cotangent, start=start, size=size)

        return (grad, None, None)

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding: Any, shard_idx: int, args: tuple
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
        device_id = mesh.devices[shard_idx] if shard_idx < len(mesh.devices) else shard_idx

        local_start_indices = []
        local_sizes = []

        for dim_idx, (g_start, g_size) in enumerate(zip(global_start, global_size)):
            dim_spec = (
                input_sharding.dim_specs[dim_idx]
                if dim_idx < len(input_sharding.dim_specs)
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

        # Reconstruct slices for maxpr (assuming full rank coverage)
        final_slices = []
        for s, sz in zip(local_start_indices, local_sizes):
            final_slices.append(slice(s, s + sz))
        
        # If there were batch dims handled by __call__ (prepended slices), 
        # we might be missing them if we just replace `slices`.
        # However, `start` and `size` usually cover the whole tensor rank for `slice_tensor`.
        # If they don't, we might need to handle it. 
        # But `slice_tensor` impl in `__call__` loops over `start`/`size`.
        # We assume `start`/`size` match rank here.
        
        new_kwargs["slices"] = tuple(final_slices)

        return new_kwargs



class ConcatenateOp(LogicalAxisOperation):
    """Concatenate tensors along an axis."""

    @property
    def name(self) -> str:
        return "concatenate"

    def __call__(self, tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
        if not tensors:
            raise ValueError("concatenate expects at least one tensor")

        return super().__call__(tensors, axis=axis)

    def maxpr(self, tensors: list[TensorValue], *, axis: int = 0) -> TensorValue:
        return ops.concat(tensors, axis)

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
        for input_idx in range(num_inputs):
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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for concatenate: split cotangent along axis."""
        from ...ops.view import slice_tensor

        axis = output.op_kwargs.get("axis", 0)
        
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
            
            sizes = [int(d) for d in cotangent.shape] # Use full shape
            sizes[axis] = dim_size # Update split axis size
            
            slc = slice_tensor(cotangent, start=starts, size=sizes)
            cotangent_slices.append(slc)
            
            start += dim_size
            
        return cotangent_slices


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

    return _broadcast_to_op(x, shape=shape)


def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _reshape_op(x, shape=shape)


_slice_update_op = SliceUpdateOp()

def slice_tensor(x: Tensor, start: Any, size: Any) -> Tensor:
    return _slice_tensor_op(x, start=start, size=size)

def slice_update(x: Tensor, update: Tensor, start: Any, size: Any) -> Tensor:
    """Update a slice of x with new values."""
    from ..base import ensure_tensor
    x = ensure_tensor(x)
    update = ensure_tensor(update)
    return _slice_update_op(x, update, start=start, size=size)


def concatenate(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    return _concatenate_op(tensors, axis=axis)


def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along a new axis."""
    expanded = [unsqueeze(t, axis=axis) for t in tensors]
    return concatenate(expanded, axis=axis)


class BroadcastToPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "broadcast_to_physical"

    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)

    def __call__(self, x: Tensor, *, shape: tuple[int, ...]) -> Tensor:
        """Broadcast physical shape, auto-incrementing batch_dims when rank increases.

        Args:
            x: Input tensor
            shape: Target GLOBAL physical shape
        """
        from .axes import unsqueeze_physical
        from .batch import incr_batch_dims

        in_rank = (
            len(x.global_shape) if x.global_shape else len(x.local_shape or x.shape)
        )
        out_rank = len(shape)
        added_dims = max(0, out_rank - in_rank)

        in_batch_dims = x.batch_dims
        for _ in range(added_dims):
            x = unsqueeze_physical(x, axis=in_batch_dims)

        result = super().__call__(x, shape=shape)

        if added_dims > 0:
            # We don't necessarily want to increment batch_dims here
            # if the added dims were for the logical part of the shape.
            # BinaryOp expects the batch_dims to match the input's.
            pass

        return result

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

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        """VJP for broadcast_to_physical: sum over broadcasted dimensions."""
        from ...ops.reduction import reduce_sum_physical

        primal_phys = tuple(
            int(d) for d in (primals.physical_global_shape or primals.local_shape)
        )
        output_phys = tuple(
            int(d) for d in (output.physical_global_shape or output.local_shape)
        )

        in_rank = len(primal_phys)
        out_rank = len(output_phys)
        offset = out_rank - in_rank

        result = cotangent
        # 1. Sum over leading new dimensions (which became batch dims)
        for _ in range(offset):
            result = reduce_sum_physical(result, axis=0, keepdims=False)

        # 2. Sum over dimensions that were size 1 and got broadcast physically
        for i in reversed(range(in_rank)):
            if primal_phys[i] == 1 and output_phys[i + offset] > 1:
                result = reduce_sum_physical(result, axis=i, keepdims=True)

        return result

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding, shard_idx: int, args: tuple
    ) -> dict:
        """Convert global target shape to local shape for each shard."""
        from ...core.sharding.spec import compute_local_shape

        global_shape = kwargs.get("shape")
        if global_shape is None or output_sharding is None:
            return kwargs

        local_shape = compute_local_shape(
            global_shape, output_sharding, device_id=shard_idx
        )
        return {**kwargs, "shape": local_shape}


__all__ = [
    "broadcast_to",
    "reshape",
    "slice_tensor",
    "concatenate",
    "stack",
    "BroadcastToPhysicalOp",
    "broadcast_to_physical",
    "SliceTensorOp",
]

_broadcast_to_physical_op = BroadcastToPhysicalOp()


def broadcast_to_physical(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _broadcast_to_physical_op(x, shape=shape)
