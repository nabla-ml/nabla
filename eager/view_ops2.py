# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Updated View Operations
# ===----------------------------------------------------------------------=== #

"""View operations with proper batch_dims handling.

Categories:
1. Physical ops (inherit Operation): Work on physical axes, no translation
2. Logical axis ops (inherit LogicalAxisOperation): Translate int kwargs
3. Logical shape ops (inherit LogicalShapeOperation): Prepend batch_shape
4. Batch management ops: Explicitly modify batch_dims
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .ops2 import Operation, LogicalAxisOperation, LogicalShapeOperation

if TYPE_CHECKING:
    from .tensor import Tensor


# =============================================================================
# Logical Axis Operations (translate ALL integer kwargs)
# =============================================================================

class UnsqueezeOp(LogicalAxisOperation):
    """Add dim at LOGICAL axis."""
    axis_offset_for_insert = True
    
    @property
    def name(self) -> str:
        return "unsqueeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)


class SqueezeOp(LogicalAxisOperation):
    """Remove dim at LOGICAL axis."""
    axis_offset_for_insert = False
    
    @property
    def name(self) -> str:
        return "squeeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)


class SwapAxesOp(LogicalAxisOperation):
    """Swap two LOGICAL axes."""
    
    @property
    def name(self) -> str:
        return "swap_axes"
    
    def maxpr(self, x: TensorValue, *, axis1: int, axis2: int) -> TensorValue:
        return ops.transpose(x, axis1, axis2)


# =============================================================================
# Logical Shape Operations (prepend batch_shape)
# =============================================================================

class BroadcastToOp(LogicalShapeOperation):
    """Broadcast to LOGICAL shape."""
    
    @property
    def name(self) -> str:
        return "broadcast_to"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)


class ReshapeOp(LogicalShapeOperation):
    """Reshape to LOGICAL shape."""
    
    @property
    def name(self) -> str:
        return "reshape"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.reshape(x, shape)


# =============================================================================
# Physical Operations (no translation, work on physical shape directly)
# =============================================================================

class MoveAxisOp(Operation):
    """Move axis in PHYSICAL shape. Preserves batch_dims."""
    
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
    """Unsqueeze at PHYSICAL axis."""
    
    @property
    def name(self) -> str:
        return "unsqueeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)


class SqueezePhysicalOp(Operation):
    """Squeeze at PHYSICAL axis."""
    
    @property
    def name(self) -> str:
        return "squeeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)


class BroadcastToPhysicalOp(Operation):
    """Broadcast to PHYSICAL shape."""
    
    @property
    def name(self) -> str:
        return "broadcast_to_physical"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)


# =============================================================================
# Batch Dims Management Operations (explicitly modify batch_dims)
# =============================================================================

def _copy_impl_with_batch_dims(x: "Tensor", new_batch_dims: int) -> "Tensor":
    """Create new Tensor with modified batch_dims."""
    from .tensor import Tensor
    from .tensor_impl import TensorImpl
    
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
    """Increment batch_dims by 1. Metadata-only."""
    
    @property
    def name(self) -> str:
        return "incr_batch_dims"
    
    def maxpr(self, x: TensorValue) -> TensorValue:
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        return _copy_impl_with_batch_dims(x, x._impl.batch_dims + 1)


class DecrBatchDimsOp(Operation):
    """Decrement batch_dims by 1. Metadata-only."""
    
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
    """Move LOGICAL axis to front of physical, increment batch_dims."""
    
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
        from .tensor import Tensor
        
        batch_dims = x._impl.batch_dims
        logical_rank = len(x.shape)
        
        if axis < 0:
            axis = logical_rank + axis
        
        physical_axis = batch_dims + axis
        result = super().__call__(x, physical_axis=physical_axis)
        result._impl.batch_dims = batch_dims + 1
        return result


class MoveAxisFromBatchDimsOp(Operation):
    """Move batch axis to LOGICAL position, decrement batch_dims."""
    
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
        from .tensor import Tensor
        
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
        result._impl.batch_dims = new_batch_dims
        return result


class BroadcastBatchDimsOp(Operation):
    """Broadcast only batch dims to target batch shape."""
    
    @property
    def name(self) -> str:
        return "broadcast_batch_dims"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.broadcast_to(x, shape)
    
    def __call__(self, x: Tensor, *, batch_shape: tuple[int, ...]) -> Tensor:
        from .tensor import Tensor
        
        logical_shape = tuple(x.shape)
        physical_shape = tuple(batch_shape) + logical_shape
        
        result = super().__call__(x, shape=physical_shape)
        result._impl.batch_dims = len(batch_shape)
        return result


# =============================================================================
# Singletons
# =============================================================================

_unsqueeze_op = UnsqueezeOp()
_squeeze_op = SqueezeOp()
_swap_axes_op = SwapAxesOp()
_broadcast_to_op = BroadcastToOp()
_reshape_op = ReshapeOp()
_moveaxis_op = MoveAxisOp()
_unsqueeze_physical_op = UnsqueezePhysicalOp()
_squeeze_physical_op = SqueezePhysicalOp()
_broadcast_to_physical_op = BroadcastToPhysicalOp()
_incr_batch_dims_op = IncrBatchDimsOp()
_decr_batch_dims_op = DecrBatchDimsOp()
_move_axis_to_batch_dims_op = MoveAxisToBatchDimsOp()
_move_axis_from_batch_dims_op = MoveAxisFromBatchDimsOp()
_broadcast_batch_dims_op = BroadcastBatchDimsOp()


# =============================================================================
# Function API
# =============================================================================

def unsqueeze(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_op(x, axis=axis)

def squeeze(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_op(x, axis=axis)

def swap_axes(x: Tensor, axis1: int, axis2: int) -> Tensor:
    return _swap_axes_op(x, axis1=axis1, axis2=axis2)

def broadcast_to(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _broadcast_to_op(x, shape=shape)

def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _reshape_op(x, shape=shape)

def moveaxis(x: Tensor, source: int, destination: int) -> Tensor:
    return _moveaxis_op(x, source=source, destination=destination)

def unsqueeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_physical_op(x, axis=axis)

def squeeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_physical_op(x, axis=axis)

def broadcast_to_physical(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _broadcast_to_physical_op(x, shape=shape)

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
    # Classes
    "UnsqueezeOp", "SqueezeOp", "SwapAxesOp",
    "BroadcastToOp", "ReshapeOp",
    "MoveAxisOp", "UnsqueezePhysicalOp", "SqueezePhysicalOp", "BroadcastToPhysicalOp",
    "IncrBatchDimsOp", "DecrBatchDimsOp", "MoveAxisToBatchDimsOp", "MoveAxisFromBatchDimsOp", "BroadcastBatchDimsOp",
    # Functions
    "unsqueeze", "squeeze", "swap_axes", "broadcast_to", "reshape", "moveaxis",
    "unsqueeze_physical", "squeeze_physical", "broadcast_to_physical",
    "incr_batch_dims", "decr_batch_dims", "move_axis_to_batch_dims", "move_axis_from_batch_dims", "broadcast_batch_dims",
]
