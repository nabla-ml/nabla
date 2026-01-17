# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from .operation import Operation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


# =============================================================================
# Physical View Ops
# =============================================================================

class MoveAxisOp(Operation):
    @property
    def name(self) -> str:
        return "moveaxis"
    
    def maxpr(self, x: TensorValue, *, source: int, destination: int) -> TensorValue:
        rank = len(x.type.shape)
        if source < 0:
            source = rank + source
        if destination < 0:
            destination = rank + destination
        
        order = list(range(rank))
        order.pop(source)
        order.insert(destination, source)
        return ops.permute(x, tuple(order))
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ..sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        source = kwargs.get("source")
        destination = kwargs.get("destination")
        
        # Normalize axes
        if source < 0: source += rank
        if destination < 0: destination += rank
        
        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)
        
        # Calculate permutation
        perm = list(factors)
        val = perm.pop(source)
        perm.insert(destination, val)
        out_str = " ".join(perm)
        
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)


class UnsqueezePhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "unsqueeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ..sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)
        
        # Insert "new_dim"
        out_factors = list(factors)
        out_factors.insert(axis, "new_dim")
        out_str = " ".join(out_factors)
                
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)
    
    def infer_output_rank(self, input_shapes: tuple[tuple[int, ...], ...], **kwargs) -> int:
        return len(input_shapes[0]) + 1


class SqueezePhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "squeeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ..sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)
        
        # Remove factor at axis
        out_factors = list(factors)
        out_factors.pop(axis)
        out_str = " ".join(out_factors)
                
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)

    def infer_output_rank(self, input_shapes: tuple[tuple[int, ...], ...], **kwargs) -> int:
        return len(input_shapes[0]) - 1


class BroadcastToPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "broadcast_to_physical"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)
    
    def __call__(self, x: "Tensor", *, shape: tuple[int, ...]) -> "Tensor":
        """Broadcast physical shape, auto-incrementing batch_dims when rank increases.
        
        Args:
            x: Input tensor
            shape: Target GLOBAL physical shape
        """
        # Use global_shape for rank comparison since target shape is global
        in_rank = len(x.global_shape) if x.global_shape else len(x.local_shape or x.shape)
        out_rank = len(shape)
        added_dims = max(0, out_rank - in_rank)
        
        # Unsqueeze at position 0 for each missing dimension (traced operation)
        for _ in range(added_dims):
            x = _unsqueeze_physical_op(x, axis=0)
        
        result = super().__call__(x, shape=shape)
        
        # Increment batch_dims for each new leading dimension added
        if added_dims > 0:
            for _ in range(added_dims):
                result = _incr_batch_dims_op(result)
        
        return result
    
    def infer_output_rank(self, input_shapes: tuple[tuple[int, ...], ...], **kwargs) -> int:
        """Output rank is the target shape's rank."""
        target_shape = kwargs.get("shape")
        if target_shape is not None:
            return len(target_shape)
        # Fallback to input rank if shape not provided
        return len(input_shapes[0]) if input_shapes else 0

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        # Custom broadcast rule: 1->N maps to [] -> [d], N->N maps to [d] -> [d]
        from ..sharding.propagation import OpShardingRuleTemplate
        
        in_shape = input_shapes[0]
        
        # Output shape source: kwargs 'shape' is primary for Physical op,
        # but output_shapes[0] is valid if provided.
        if output_shapes is not None:
             out_shape = output_shapes[0]
        else:
             out_shape = kwargs.get("shape")
             
        if out_shape is None:
             raise ValueError("BroadcastToPhysicalOp requires 'shape' kwarg or output_shapes")
        
        in_rank = len(in_shape)
        out_rank = len(out_shape)        
        offset = out_rank - in_rank
        
        factors = [f"d{i}" for i in range(out_rank)]
        out_mapping = {i: [factors[i]] for i in range(out_rank)}
        in_mapping = {}
        
        for i in range(in_rank):
            # Input dim i corresponds to Output dim i + offset
            out_dim_idx = i + offset
            in_dim_size = in_shape[i]
            out_dim_size = out_shape[out_dim_idx]
            
            if in_dim_size == 1 and out_dim_size > 1:
                in_mapping[i] = []
            else:
                in_mapping[i] = [factors[out_dim_idx]]
                
        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(input_shapes, output_shapes)

    def _transform_shard_kwargs(self, kwargs: dict, output_sharding: Any, shard_idx: int) -> dict:
        """Convert global target shape to local shape for each shard."""
        from ..sharding.spec import compute_local_shape
        
        global_shape = kwargs.get('shape')
        if global_shape is None or output_sharding is None:
            return kwargs
        
        # output_sharding is expected to be a single ShardingSpec for this op
        local_shape = compute_local_shape(global_shape, output_sharding, device_id=shard_idx)
        return {**kwargs, 'shape': local_shape}


# =============================================================================
# Physical Reduction Ops
# =============================================================================

class ReduceSumPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "reduce_sum_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int, keepdims: bool = False) -> TensorValue:
        # maxpr must only have ONE MAX operation for sharding propagation to work correctly.
        return ops.sum(x, axis=axis)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce: (d0, d1, ...) -> (d0, 1, ...) with reduce_dim kept as size 1."""
        from ..sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)
        
        out_factors = list(factors)
        if 0 <= axis < rank:
            out_factors[axis] = "1"
        out_str = " ".join(out_factors)
        
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class MeanPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "mean_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int, keepdims: bool = False) -> TensorValue:
        # maxpr must only have ONE MAX operation for sharding propagation to work correctly.
        return ops.mean(x, axis=axis)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce: (d0, d1, ...) -> (d0, 1, ...) with reduce_dim kept as size 1."""
        from ..sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)
        
        out_factors = list(factors)
        if 0 <= axis < rank:
            out_factors[axis] = "1"
        out_str = " ".join(out_factors)
        
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]
        if axis < 0:
            axis = len(in_shape) + axis
        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


# =============================================================================
# Batch Management Ops (Explicit Metadata Modification)
# =============================================================================

def _copy_impl_with_batch_dims(x: "Tensor", new_batch_dims: int, op: "Operation" = None, kwargs: dict = None) -> "Tensor":
    from ..core import Tensor
    from ..core import TensorImpl
    
    new_impl = TensorImpl(
        storages=x._impl._storages,
        values=x._impl._values,
        traced=x._impl.traced,
        batch_dims=new_batch_dims,
    )
    # Sharding must be set after construction (not a constructor arg)
    new_impl.sharding = x._impl.sharding

    
    output = Tensor(impl=new_impl)
    
    # Setup tracing refs if op provided
    if op is not None and x._impl.traced:
        op._setup_output_refs(output, (x,), kwargs or {}, True)
    
    return output


class IncrBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "incr_batch_dims"
    
    def maxpr(self, x: TensorValue) -> TensorValue:
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        return _copy_impl_with_batch_dims(x, x._impl.batch_dims + 1, op=self, kwargs={})


class DecrBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "decr_batch_dims"
    
    def maxpr(self, x: TensorValue) -> TensorValue:
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        if x._impl.batch_dims <= 0:
            raise ValueError("Cannot decrement batch_dims below 0")
        return _copy_impl_with_batch_dims(x, x._impl.batch_dims - 1, op=self, kwargs={})


class MoveAxisToBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "move_axis_to_batch_dims"
    
    def maxpr(self, x: TensorValue, *, physical_axis: int) -> TensorValue:
        rank = len(x.type.shape)
        order = list(range(rank))
        order.pop(physical_axis)
        order.insert(0, physical_axis)
        return ops.permute(x, tuple(order))
    
    def __call__(self, x: Tensor, *, axis: int) -> Tensor:
        batch_dims = x._impl.batch_dims
        logical_rank = len(x.shape)
        
        if axis < 0:
            axis = logical_rank + axis
        
        physical_axis = batch_dims + axis
        result = super().__call__(x, physical_axis=physical_axis)
        return _copy_impl_with_batch_dims(result, batch_dims + 1)


class MoveAxisFromBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "move_axis_from_batch_dims"
    
    def maxpr(self, x: TensorValue, *, physical_source: int, physical_destination: int) -> TensorValue:
        rank = len(x.type.shape)
        order = list(range(rank))
        order.pop(physical_source)
        order.insert(physical_destination, physical_source)
        return ops.permute(x, tuple(order))
    
    def __call__(self, x: Tensor, *, batch_axis: int = 0, logical_destination: int = 0) -> Tensor:
        current_batch_dims = x._impl.batch_dims
        if current_batch_dims <= 0:
            raise ValueError("No batch dims to move from")
        
        logical_rank = len(x.shape)
        
        if batch_axis < 0:
            batch_axis = current_batch_dims + batch_axis
        
        physical_source = batch_axis
        new_batch_dims = current_batch_dims - 1
        new_logical_rank = logical_rank + 1
        
        if logical_destination < 0:
            logical_destination = new_logical_rank + logical_destination
        
        physical_destination = new_batch_dims + logical_destination
        
        result = super().__call__(x, physical_source=physical_source, physical_destination=physical_destination)
        return _copy_impl_with_batch_dims(result, new_batch_dims)


class BroadcastBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "broadcast_batch_dims"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)
    
    def __call__(self, x: Tensor, *, batch_shape: tuple[int, ...]) -> Tensor:
        logical_shape = tuple(x.shape)
        physical_shape = tuple(batch_shape) + logical_shape
        
        result = super().__call__(x, shape=physical_shape)
        return _copy_impl_with_batch_dims(result, len(batch_shape))


# =============================================================================
# Functions
# =============================================================================

_moveaxis_op = MoveAxisOp()
_unsqueeze_physical_op = UnsqueezePhysicalOp()
_squeeze_physical_op = SqueezePhysicalOp()
_broadcast_to_physical_op = BroadcastToPhysicalOp()
_reduce_sum_physical_op = ReduceSumPhysicalOp()
_mean_physical_op = MeanPhysicalOp()
_incr_batch_dims_op = IncrBatchDimsOp()
_decr_batch_dims_op = DecrBatchDimsOp()
_move_axis_to_batch_dims_op = MoveAxisToBatchDimsOp()
_move_axis_from_batch_dims_op = MoveAxisFromBatchDimsOp()
_broadcast_batch_dims_op = BroadcastBatchDimsOp()

def moveaxis(x: Tensor, source: int, destination: int) -> Tensor:
    return _moveaxis_op(x, source=source, destination=destination)

def unsqueeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_physical_op(x, axis=axis)

def squeeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_physical_op(x, axis=axis)

def broadcast_to_physical(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _broadcast_to_physical_op(x, shape=shape)

def reduce_sum_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    # maxpr always keeps dims; squeeze at Tensor level so sharding propagation handles it
    result = _reduce_sum_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result

def mean_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    # maxpr always keeps dims; squeeze at Tensor level so sharding propagation handles it
    result = _mean_physical_op(x, axis=axis, keepdims=True)
    if not keepdims:
        result = _squeeze_physical_op(result, axis=axis)
    return result

def incr_batch_dims(x: Tensor) -> Tensor:
    return _incr_batch_dims_op(x)

def decr_batch_dims(x: Tensor) -> Tensor:
    return _decr_batch_dims_op(x)

def move_axis_to_batch_dims(x: Tensor, axis: int) -> Tensor:
    return _move_axis_to_batch_dims_op(x, axis=axis)

def move_axis_from_batch_dims(x: Tensor, batch_axis: int = 0, logical_destination: int = 0) -> Tensor:
    return _move_axis_from_batch_dims_op(x, batch_axis=batch_axis, logical_destination=logical_destination)

def broadcast_batch_dims(x: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    return _broadcast_batch_dims_op(x, batch_shape=batch_shape)


__all__ = [
    "MoveAxisOp", "UnsqueezePhysicalOp", "SqueezePhysicalOp", "BroadcastToPhysicalOp",
    "ReduceSumPhysicalOp", "MeanPhysicalOp",
    "IncrBatchDimsOp", "DecrBatchDimsOp", "MoveAxisToBatchDimsOp", "MoveAxisFromBatchDimsOp", "BroadcastBatchDimsOp",
    "moveaxis", "unsqueeze_physical", "squeeze_physical", "broadcast_to_physical",
    "reduce_sum_physical", "mean_physical",
    "incr_batch_dims", "decr_batch_dims", "move_axis_to_batch_dims", "move_axis_from_batch_dims", "broadcast_batch_dims",
]
