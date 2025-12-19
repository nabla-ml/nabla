# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Physical Operations
# ===----------------------------------------------------------------------=== #

"""Physical operations working directly on the underlying tensor shape.

These operations do NOT apply any translation logic.
This includes:
- View ops (moveaxis, physical unsqueeze/squeeze)
- Reductions (sum/mean over physical axes)
- Batch dimension management (modifying metadata)
"""

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


class UnsqueezePhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "unsqueeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)


class SqueezePhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "squeeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)


class BroadcastToPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "broadcast_to_physical"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)


# =============================================================================
# Physical Reduction Ops
# =============================================================================

class ReduceSumPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "reduce_sum_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int, keepdims: bool = False) -> TensorValue:
        result = ops.sum(x, axis=axis)
        if not keepdims:
            result = ops.squeeze(result, axis=axis)
        return result


class MeanPhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "mean_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int, keepdims: bool = False) -> TensorValue:
        result = ops.mean(x, axis=axis)
        if not keepdims:
            result = ops.squeeze(result, axis=axis)
        return result


# =============================================================================
# Batch Management Ops (Explicit Metadata Modification)
# =============================================================================

def _copy_impl_with_batch_dims(x: "Tensor", new_batch_dims: int) -> "Tensor":
    from ..core.tensor import Tensor
    from ..core.tensor_impl import TensorImpl
    
    new_impl = TensorImpl(
        storages=x._impl._storages,
        values=x._impl._values,
        traced=x._impl.traced,
        batch_dims=new_batch_dims,
        sharding=x._impl.sharding,
    )
    new_impl.cached_shape = x._impl.cached_shape
    new_impl.cached_dtype = x._impl.cached_dtype
    new_impl.cached_device = x._impl.cached_device
    return Tensor(impl=new_impl)


class IncrBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "incr_batch_dims"
    
    def maxpr(self, x: TensorValue) -> TensorValue:
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        return _copy_impl_with_batch_dims(x, x._impl.batch_dims + 1)


class DecrBatchDimsOp(Operation):
    @property
    def name(self) -> str:
        return "decr_batch_dims"
    
    def maxpr(self, x: TensorValue) -> TensorValue:
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        if x._impl.batch_dims <= 0:
            raise ValueError("Cannot decrement batch_dims below 0")
        return _copy_impl_with_batch_dims(x, x._impl.batch_dims - 1)


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
    return _reduce_sum_physical_op(x, axis=axis, keepdims=keepdims)

def mean_physical(x: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return _mean_physical_op(x, axis=axis, keepdims=keepdims)

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
