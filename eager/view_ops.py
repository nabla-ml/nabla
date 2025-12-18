# ===----------------------------------------------------------------------=== #
# Nabla 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""View operations for the eager module.

These operations change tensor shape/layout without copying data.
Focus on forward logic for vmap support - vjp/jvp rules to be added later.

Key operations for vmap:
- unsqueeze/squeeze: Add/remove dimensions
- swap_axes/moveaxis: Reorder dimensions  
- broadcast_to: Explicit broadcasting
- move_axis_to_batch_dims/move_axis_from_batch_dims: Move axes between batch/logical
- incr_batch_dims/decr_batch_dims: Adjust batch_dims counter
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .ops import Operation

if TYPE_CHECKING:
    from .tensor import Tensor


# =============================================================================
# Unsqueeze / Squeeze Operations  
# =============================================================================
#
# Two variants of each operation:
# - UnsqueezeOp/SqueezeOp: User-facing, operates on LOGICAL axes
#   (inherits LogicalAxisOperation which handles axis translation)
# - UnsqueezePhysicalOp/SqueezePhysicalOp: Internal ops for vmap,
#   operates directly on PHYSICAL axes (inherits base Operation)

from .ops import LogicalAxisOperation, LogicalShapeOperation


class UnsqueezeOp(LogicalAxisOperation):
    """Add a dimension of size 1 at the specified LOGICAL axis.
    
    Automatically translates logical axis to physical axis by adding batch_dims offset.
    Preserves batch_dims on output.
    
    Args:
        axis: Logical axis position (supports negative indexing)
        
    Example:
        # logical shape (8,) with batch_dims=1 -> physical (4, 8)
        # unsqueeze(axis=0) -> logical (1, 8) -> physical (4, 1, 8)
    """
    
    axis_offset_for_insert = True  # For unsqueeze: negative axis needs +1
    
    @property
    def name(self) -> str:
        return "unsqueeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        """Add dimension at the given physical axis position."""
        return ops.unsqueeze(x, axis)


class SqueezeOp(LogicalAxisOperation):
    """Remove a dimension of size 1 at the specified LOGICAL axis.
    
    Automatically translates logical axis to physical axis by adding batch_dims offset.
    Preserves batch_dims on output.
    
    Args:
        axis: Logical axis position to squeeze (supports negative indexing, must be size 1)
        
    Example:
        # logical shape (1, 8) with batch_dims=1 -> physical (4, 1, 8)
        # squeeze(axis=0) -> logical (8,) -> physical (4, 8)
    """
    
    axis_offset_for_insert = False  # For squeeze: no offset needed
    
    @property
    def name(self) -> str:
        return "squeeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        """Remove dimension at the given physical axis position."""
        return ops.squeeze(x, axis)


class UnsqueezePhysicalOp(Operation):
    """Add a dimension of size 1 at the specified PHYSICAL axis.
    
    Internal operation for vmap. Does NOT translate axis.
    Does NOT preserve batch_dims - caller must manage.
    """
    
    @property
    def name(self) -> str:
        return "unsqueeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        """Add dimension at the given physical axis position."""
        return ops.unsqueeze(x, axis)


class SqueezePhysicalOp(Operation):
    """Remove a dimension of size 1 at the specified PHYSICAL axis.
    
    Internal operation for vmap. Does NOT translate axis.
    Does NOT preserve batch_dims - caller must manage.
    """
    
    @property
    def name(self) -> str:
        return "squeeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        """Remove dimension at the given physical axis position."""
        return ops.squeeze(x, axis)


# =============================================================================
# Axis Movement Operations
# =============================================================================

class SwapAxesOp(Operation):
    """Swap two axes of a tensor.
    
    Equivalent to transpose with two specific axes.
    Does not affect batch_dims counter.
    """
    
    @property
    def name(self) -> str:
        return "swap_axes"
    
    def maxpr(self, x: TensorValue, *, axis1: int, axis2: int) -> TensorValue:
        """Swap axis1 and axis2."""
        rank = len(x.type.shape)
        
        # Normalize negative indices
        if axis1 < 0:
            axis1 = rank + axis1
        if axis2 < 0:
            axis2 = rank + axis2
        
        # MAX's transpose takes two axes to swap
        return ops.transpose(x, axis1, axis2)


class MoveAxisOp(Operation):
    """Move an axis from one position to another.
    
    Similar to numpy.moveaxis for a single axis.
    Does not affect batch_dims counter.
    """
    
    @property
    def name(self) -> str:
        return "moveaxis"
    
    def maxpr(self, x: TensorValue, *, source: int, destination: int) -> TensorValue:
        """Move axis from source to destination position."""
        rank = len(x.type.shape)
        
        # Normalize negative indices
        if source < 0:
            source = rank + source
        if destination < 0:
            destination = rank + destination
        
        # Build permutation
        order = list(range(rank))
        order.pop(source)
        order.insert(destination, source)
        
        # Use ops.permute for full permutation
        return ops.permute(x, tuple(order))


# =============================================================================
# Broadcast and Reshape Operations
# =============================================================================
#
# Both operations take a shape parameter on LOGICAL shape and need to
# preserve batch_dims. They inherit from LogicalShapeOperation.


class BroadcastToOp(LogicalShapeOperation):
    """Broadcast tensor to a target LOGICAL shape.
    
    Automatically prepends batch_shape to create physical shape.
    Preserves batch_dims on output.
    
    Uses numpy broadcasting rules.
    
    Example:
        # logical shape (3, 1) with batch_dims=1 -> physical (4, 3, 1)
        # broadcast_to(shape=(3, 5)) -> logical (3, 5) -> physical (4, 3, 5)
    """
    
    @property
    def name(self) -> str:
        return "broadcast_to"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        """Broadcast x to target physical shape."""
        return ops.broadcast_to(x, shape)


class ReshapeOp(LogicalShapeOperation):
    """Reshape tensor to a new LOGICAL shape.
    
    Automatically prepends batch_shape to create physical shape.
    Preserves batch_dims on output.
    
    Example:
        # Physical: (batch=5, 12), batch_dims=1, logical: (12,)
        y = reshape(x, shape=(3, 4))
        # Physical: (batch=5, 3, 4), batch_dims=1, logical: (3, 4)
    """
    
    @property
    def name(self) -> str:
        return "reshape"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        """Reshape x to target physical shape."""
        return ops.reshape(x, shape)


class BroadcastBatchDimsOp(Operation):
    """Broadcast only the BATCH dimensions to a target batch shape.
    
    Keeps the logical shape unchanged, only broadcasts the batch prefix.
    Updates batch_dims to match the new batch shape length.
    
    This is useful for aligning batch dimensions between tensors
    before operations like matmul.
    
    Example:
        # Input: physical (1, 8, 4), batch_dims=1, logical (8, 4)
        # broadcast_batch_dims(batch_shape=(5,))
        # Output: physical (5, 8, 4), batch_dims=1, logical (8, 4)
        
        # Input: physical (8, 4), batch_dims=0, logical (8, 4)
        # broadcast_batch_dims(batch_shape=(5, 3))
        # Output: physical (5, 3, 8, 4), batch_dims=2, logical (8, 4)
    """
    
    @property
    def name(self) -> str:
        return "broadcast_batch_dims"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        """Broadcast x to target physical shape."""
        return ops.broadcast_to(x, shape)
    
    def __call__(self, x: "Tensor", *, batch_shape: tuple[int, ...]) -> "Tensor":
        """Broadcast batch dimensions to target batch shape.
        
        Args:
            x: Input tensor
            batch_shape: Target batch shape
            
        Returns:
            Tensor with new batch_shape, same logical shape
        """
        from .tensor import Tensor
        
        logical_shape = tuple(x.shape)
        
        # Build physical shape = target batch_shape + logical_shape
        physical_shape = tuple(batch_shape) + logical_shape
        
        # Call parent which passes physical shape to maxpr
        result = super().__call__(x, shape=physical_shape)
        
        # Set batch_dims to match new batch shape
        if isinstance(result, Tensor):
            result._impl.batch_dims = len(batch_shape)
        
        return result


class BroadcastToPhysicalOp(Operation):
    """Broadcast tensor to a target PHYSICAL shape.
    
    Internal operation for vmap. Does NOT translate shape.
    Does NOT preserve batch_dims - caller must manage.
    """
    
    @property
    def name(self) -> str:
        return "broadcast_to_physical"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        """Broadcast x to target physical shape."""
        return ops.broadcast_to(x, shape)


# =============================================================================
# Batch Dims Management Operations
# =============================================================================


def _copy_impl_with_batch_dims(x: "Tensor", new_batch_dims: int) -> "Tensor":
    """Create a new Tensor with modified batch_dims (metadata-only operation).
    
    This helper extracts the common pattern of creating a new TensorImpl
    with a different batch_dims count while preserving all other state.
    
    Args:
        x: Source tensor
        new_batch_dims: New value for batch_dims counter
        
    Returns:
        New Tensor with updated batch_dims
    """
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
    """Increment the batch_dims counter by 1.
    
    This is a metadata-only operation - no change to underlying data.
    Used by vmap to mark leading dims as batch dims.
    """
    
    @property
    def name(self) -> str:
        return "incr_batch_dims"
    
    def maxpr(self, x: TensorValue) -> TensorValue:
        """Identity - no graph change, just metadata."""
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        """Increment batch_dims and return new tensor."""
        return _copy_impl_with_batch_dims(x, x._impl.batch_dims + 1)


class DecrBatchDimsOp(Operation):
    """Decrement the batch_dims counter by 1.
    
    This is a metadata-only operation - no change to underlying data.
    Used by vmap to unmark batch dims.
    """
    
    @property
    def name(self) -> str:
        return "decr_batch_dims"
    
    def maxpr(self, x: TensorValue) -> TensorValue:
        """Identity - no graph change, just metadata."""
        return x
    
    def __call__(self, x: Tensor) -> Tensor:
        """Decrement batch_dims and return new tensor."""
        if x._impl.batch_dims <= 0:
            raise ValueError("Cannot decrement batch_dims below 0")
        return _copy_impl_with_batch_dims(x, x._impl.batch_dims - 1)


class MoveAxisToBatchDimsOp(Operation):
    """Move an axis from logical shape to batch dims.
    
    This operation:
    1. Takes a LOGICAL axis index (user's view, unaware of existing batch dims)
    2. Translates to physical axis by adding batch_dims offset
    3. Moves that physical axis to position 0 (front of batch dims)
    4. Increments batch_dims counter
    
    Used by vmap to convert a logical axis into a batch axis.
    
    Example:
        Input: physical=(B, H, W, C) = (2, 3, 4, 5), batch_dims=1
               logical=(H, W, C) = (3, 4, 5)
        
        move_axis_to_batch_dims(x, axis=2)  # axis=2 is C in logical
        
        physical_axis = 1 + 2 = 3  (C is at physical position 3)
        Output: physical=(C, B, H, W) = (5, 2, 3, 4), batch_dims=2
                logical=(H, W) = (3, 4)
    """
    
    @property
    def name(self) -> str:
        return "move_axis_to_batch_dims"
    
    def maxpr(self, x: TensorValue, *, physical_axis: int) -> TensorValue:
        """Move physical_axis to front. Called internally with translated axis."""
        rank = len(x.type.shape)
        
        # Move to front
        order = list(range(rank))
        order.pop(physical_axis)
        order.insert(0, physical_axis)
        
        return ops.permute(x, tuple(order))
    
    def __call__(self, x: Tensor, *, axis: int) -> Tensor:
        """Move logical axis to batch dims and increment counter.
        
        Args:
            x: Input tensor
            axis: LOGICAL axis index (relative to user-visible shape)
        """
        from .tensor import Tensor
        from .tensor_impl import TensorImpl
        
        batch_dims = x._impl.batch_dims
        logical_rank = len(x.shape)  # This is logical rank
        
        # Normalize negative axis (relative to logical shape)
        if axis < 0:
            axis = logical_rank + axis
        
        if axis < 0 or axis >= logical_rank:
            raise ValueError(f"axis {axis} out of bounds for logical shape of rank {logical_rank}")
        
        # Translate logical axis to physical axis
        physical_axis = batch_dims + axis
        
        # Call parent's __call__ with the physical axis
        # We pass physical_axis as kwarg which maxpr expects
        result = super().__call__(x, physical_axis=physical_axis)
        
        # Increment batch_dims on result
        result._impl.batch_dims = batch_dims + 1
        
        return result


class MoveAxisFromBatchDimsOp(Operation):
    """Move an axis from batch dims to logical shape.
    
    This operation:
    1. Takes batch_axis (index within batch dims, 0 = outermost)
    2. Takes logical_destination (where in logical shape to place it)
    3. Translates to physical indices and performs the move
    4. Decrements batch_dims counter
    
    Used by vmap to convert a batch axis back to logical.
    
    Example:
        Input: physical=(C, B, H, W) = (2, 5, 3, 4), batch_dims=2
               logical=(H, W) = (3, 4)
        
        move_axis_from_batch_dims(x, batch_axis=0, logical_destination=2)
        
        # batch_axis=0 means C (physical index 0)
        # logical_destination=2 means end of new logical shape
        # new_batch_dims = 1, new logical will have rank 3
        # physical_destination = 1 + 2 = 3
        
        Output: physical=(B, H, W, C) = (5, 3, 4, 2), batch_dims=1
                logical=(H, W, C) = (3, 4, 2)
    """
    
    @property
    def name(self) -> str:
        return "move_axis_from_batch_dims"
    
    def maxpr(
        self, 
        x: TensorValue, 
        *, 
        physical_source: int,
        physical_destination: int
    ) -> TensorValue:
        """Move axis from physical_source to physical_destination.
        
        physical_destination is the FINAL position in the output tensor.
        This is called internally with already-translated physical indices.
        """
        rank = len(x.type.shape)
        
        # Build permutation: this is equivalent to numpy.moveaxis
        order = list(range(rank))
        order.pop(physical_source)
        # Insert directly at physical_destination
        # After pop, if source < destination, the indices after source shift down by 1
        # So inserting at physical_destination in the popped list gives the right result
        order.insert(physical_destination, physical_source)
        
        return ops.permute(x, tuple(order))
    
    def __call__(
        self, 
        x: Tensor, 
        *, 
        batch_axis: int = 0,
        logical_destination: int = 0
    ) -> Tensor:
        """Move axis from batch dims to logical shape.
        
        Args:
            x: Input tensor
            batch_axis: Which batch axis to move (0 = outermost batch dim)
            logical_destination: Where in LOGICAL shape to place it (0 = first logical dim)
        """
        from .tensor import Tensor
        
        current_batch_dims = x._impl.batch_dims
        
        if current_batch_dims <= 0:
            raise ValueError("No batch dims to move from")
        
        physical_shape = x._impl.physical_shape
        rank = len(physical_shape)
        logical_rank = rank - current_batch_dims
        
        # Normalize batch_axis (relative to batch dims)
        if batch_axis < 0:
            batch_axis = current_batch_dims + batch_axis
        
        if batch_axis < 0 or batch_axis >= current_batch_dims:
            raise ValueError(f"batch_axis {batch_axis} out of bounds for batch_dims={current_batch_dims}")
        
        # batch_axis IS the physical source (it's within the batch prefix)
        physical_source = batch_axis
        
        # New batch_dims after moving one out
        new_batch_dims = current_batch_dims - 1
        
        # New logical rank after adding one axis
        new_logical_rank = logical_rank + 1
        
        # Normalize logical_destination (relative to new logical shape)
        if logical_destination < 0:
            logical_destination = new_logical_rank + logical_destination
        
        if logical_destination < 0 or logical_destination > new_logical_rank:
            raise ValueError(f"logical_destination {logical_destination} out of bounds for new logical rank {new_logical_rank}")
        
        # Translate to physical destination
        # After batch_dims becomes new_batch_dims, logical starts at new_batch_dims
        physical_destination = new_batch_dims + logical_destination
        
        # Call parent's __call__ with physical indices
        result = super().__call__(
            x, 
            physical_source=physical_source,
            physical_destination=physical_destination
        )
        
        # Set new batch_dims on result
        if isinstance(result, Tensor):
            result._impl.batch_dims = new_batch_dims
        
        return result


# =============================================================================
# Singleton Instances (function-style API)
# =============================================================================

_unsqueeze_op = UnsqueezeOp()
_unsqueeze_physical_op = UnsqueezePhysicalOp()
_squeeze_op = SqueezeOp()
_squeeze_physical_op = SqueezePhysicalOp()
_swap_axes_op = SwapAxesOp()
_moveaxis_op = MoveAxisOp()
_broadcast_to_op = BroadcastToOp()
_broadcast_batch_dims_op = BroadcastBatchDimsOp()
_broadcast_to_physical_op = BroadcastToPhysicalOp()
_reshape_op = ReshapeOp()
_incr_batch_dims_op = IncrBatchDimsOp()
_decr_batch_dims_op = DecrBatchDimsOp()
_move_axis_to_batch_dims_op = MoveAxisToBatchDimsOp()
_move_axis_from_batch_dims_op = MoveAxisFromBatchDimsOp()


def unsqueeze(x: Tensor, axis: int = 0) -> Tensor:
    """Add a dimension of size 1 at the specified axis.
    
    Args:
        x: Input tensor
        axis: Position to insert new dimension (default: 0)
        
    Returns:
        Tensor with shape expanded by 1 at axis
    """
    return _unsqueeze_op(x, axis=axis)


def squeeze(x: Tensor, axis: int = 0) -> Tensor:
    """Remove a dimension of size 1 at the specified axis.
    
    Args:
        x: Input tensor
        axis: Position to remove (must be size 1)
        
    Returns:
        Tensor with dimension removed at axis
    """
    return _squeeze_op(x, axis=axis)


def unsqueeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    """Add dimension at PHYSICAL axis (for vmap internals)."""
    return _unsqueeze_physical_op(x, axis=axis)


def squeeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    """Remove dimension at PHYSICAL axis (for vmap internals)."""
    return _squeeze_physical_op(x, axis=axis)


def swap_axes(x: Tensor, axis1: int, axis2: int) -> Tensor:
    """Swap two axes of a tensor.
    
    Args:
        x: Input tensor
        axis1: First axis
        axis2: Second axis
        
    Returns:
        Tensor with axis1 and axis2 swapped
    """
    return _swap_axes_op(x, axis1=axis1, axis2=axis2)


def moveaxis(x: Tensor, source: int, destination: int) -> Tensor:
    """Move an axis from source to destination position.
    
    Args:
        x: Input tensor
        source: Current axis position
        destination: Target axis position
        
    Returns:
        Tensor with axis moved
    """
    return _moveaxis_op(x, source=source, destination=destination)


def broadcast_to(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Broadcast tensor to target LOGICAL shape.
    
    Automatically prepends batch_shape to create physical shape.
    
    Args:
        x: Input tensor
        shape: Target logical shape (must be broadcast-compatible)
        
    Returns:
        Tensor broadcasted to shape
    """
    return _broadcast_to_op(x, shape=shape)


def broadcast_batch_dims(x: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    """Broadcast only batch dimensions to target batch shape.
    
    Keeps the logical shape unchanged, only broadcasts the batch prefix.
    
    Args:
        x: Input tensor
        batch_shape: Target batch shape
        
    Returns:
        Tensor with new batch_shape, same logical shape
    """
    return _broadcast_batch_dims_op(x, batch_shape=batch_shape)


def broadcast_to_physical(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Broadcast tensor to target PHYSICAL shape (for vmap internals)."""
    return _broadcast_to_physical_op(x, shape=shape)


def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape tensor to new LOGICAL shape.
    
    Automatically prepends batch_shape to create physical shape.
    
    Args:
        x: Input tensor
        shape: Target logical shape (can include -1 for inference)
        
    Returns:
        Tensor with new logical shape, batch dims unchanged
    """
    return _reshape_op(x, shape=shape)


def incr_batch_dims(x: Tensor) -> Tensor:
    """Increment batch_dims counter by 1.
    
    Metadata-only operation for vmap.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with batch_dims += 1
    """
    return _incr_batch_dims_op(x)


def decr_batch_dims(x: Tensor) -> Tensor:
    """Decrement batch_dims counter by 1.
    
    Metadata-only operation for vmap.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with batch_dims -= 1
    """
    return _decr_batch_dims_op(x)


def move_axis_to_batch_dims(x: Tensor, axis: int) -> Tensor:
    """Move axis from logical shape to batch dims.
    
    Moves axis to front and increments batch_dims.
    
    Args:
        x: Input tensor
        axis: Axis to move (in physical coordinates)
        
    Returns:
        Tensor with axis moved to front and batch_dims += 1
    """
    return _move_axis_to_batch_dims_op(x, axis=axis)


def move_axis_from_batch_dims(
    x: Tensor, 
    batch_axis: int = 0,
    logical_destination: int = 0
) -> Tensor:
    """Move axis from batch dims to logical shape.
    
    Moves batch axis to logical destination and decrements batch_dims.
    
    Args:
        x: Input tensor
        batch_axis: Which batch axis to move (0 = outermost, default: 0)
        logical_destination: Where in logical shape to place it (default: 0)
        
    Returns:
        Tensor with axis moved from batch to logical and batch_dims -= 1
        
    Example:
        Input: physical=(2, 5, 3, 4), batch_dims=2, logical=(3, 4)
        move_axis_from_batch_dims(x, batch_axis=0, logical_destination=2)
        Output: physical=(5, 3, 4, 2), batch_dims=1, logical=(3, 4, 2)
    """
    return _move_axis_from_batch_dims_op(
        x, 
        batch_axis=batch_axis, 
        logical_destination=logical_destination
    )


__all__ = [
    # User-facing classes (logical axis/shape operations)
    "UnsqueezeOp",
    "SqueezeOp",
    "SwapAxesOp",
    "MoveAxisOp",
    "BroadcastToOp",
    "BroadcastBatchDimsOp",
    "ReshapeOp",
    "IncrBatchDimsOp",
    "DecrBatchDimsOp",
    "MoveAxisToBatchDimsOp",
    "MoveAxisFromBatchDimsOp",
    # Internal classes (physical axis/shape operations, for vmap)
    "UnsqueezePhysicalOp",
    "SqueezePhysicalOp",
    "BroadcastToPhysicalOp",
    # User-facing functions
    "unsqueeze",
    "squeeze",
    "swap_axes",
    "moveaxis",
    "broadcast_to",
    "broadcast_batch_dims",
    "reshape",
    "incr_batch_dims",
    "decr_batch_dims",
    "move_axis_to_batch_dims",
    "move_axis_from_batch_dims",
    # Internal functions (physical Operations)
    "unsqueeze_physical",
    "squeeze_physical",
    "broadcast_to_physical",
]
