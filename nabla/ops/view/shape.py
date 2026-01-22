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

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(kwargs.get("shape", input_shapes[0]))

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding, shard_idx: int
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
        if isinstance(primals, tuple):
            x = primals[0]
        else:
            x = primals
        from ...ops.reduction import reduce_sum
        
        input_shape = tuple(x.shape)
        output_shape = tuple(cotangent.shape)
        
        # Sum over new/broadcasted dimensions
        result = cotangent
        input_rank = len(input_shape)
        output_rank = len(output_shape)
        
        # Sum over leading new dimensions
        for _ in range(output_rank - input_rank):
            result = reduce_sum(result, axis=0, keepdims=False)
        
        # Sum over dimensions that were size 1 and got broadcast
        for i, (in_dim, out_dim) in enumerate(zip(input_shape, result.shape)):
            if in_dim == 1 and out_dim > 1:
                result = reduce_sum(result, axis=i, keepdims=True)
        
        return result

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for broadcast_to: broadcast the tangent."""
        if isinstance(tangents, tuple):
            t = tangents[0]
        else:
            t = tangents
        # Get target shape from output
        target_shape = tuple(output.shape)
        return broadcast_to(t, target_shape)


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
        self, kwargs: dict, output_sharding, shard_idx: int
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
        if isinstance(primals, tuple):
            x = primals[0]
        else:
            x = primals
        return reshape(cotangent, tuple(x.shape))

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """JVP for reshape: reshape tangent to output shape."""
        if isinstance(tangents, tuple):
            t = tangents[0]
        else:
            t = tangents
        target_shape = tuple(output.shape)
        return reshape(t, target_shape)


class SliceTensorOp(Operation):
    @property
    def name(self) -> str:
        return "slice_tensor"

    def maxpr(self, x: TensorValue, start: Any, size: Any) -> TensorValue:
        slices = []
        for s, sz in zip(start, size, strict=False):
            end = s + sz
            slices.append(slice(s, end))
        return ops.slice_tensor(x, slices)

    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])


class ConcatenateOp(LogicalAxisOperation):
    """Concatenate tensors along an axis."""

    @property
    def name(self) -> str:
        return "concatenate"

    def __call__(self, tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
        if not tensors:
            raise ValueError("concatenate expects at least one tensor")

        first = tensors[0]
        batch_dims = first.batch_dims

        if axis < 0:
            axis += len(first.shape)
        phys_axis = batch_dims + axis

        return super(LogicalAxisOperation, self).__call__(tensors, axis=phys_axis)

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


def slice_tensor(x: Tensor, start: Any, size: Any) -> Tensor:
    return _slice_tensor_op(x, start=start, size=size)


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

        for _ in range(added_dims):
            x = unsqueeze_physical(x, axis=0)

        result = super().__call__(x, shape=shape)

        if added_dims > 0:
            for _ in range(added_dims):
                result = incr_batch_dims(result)

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

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding: Any, shard_idx: int
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
