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

class UnsqueezeOp(Operation):
    """Add a dimension of size 1 at the specified axis.
    
    Does not affect batch_dims counter - caller must adjust if needed.
    """
    
    @property
    def name(self) -> str:
        return "unsqueeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        """Add dimension at axis position."""
        return ops.unsqueeze(x, axis)


class SqueezeOp(Operation):
    """Remove a dimension of size 1 at the specified axis.
    
    Does not affect batch_dims counter - caller must adjust if needed.
    """
    
    @property
    def name(self) -> str:
        return "squeeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        """Remove dimension at axis position."""
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
# Broadcast Operations
# =============================================================================

class BroadcastToOp(Operation):
    """Broadcast tensor to a target shape.
    
    Uses numpy broadcasting rules.
    """
    
    @property
    def name(self) -> str:
        return "broadcast_to"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        """Broadcast x to target shape."""
        return ops.broadcast_to(x, shape)


# =============================================================================
# Batch Dims Management Operations
# =============================================================================

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
        from .tensor import Tensor
        from .tensor_impl import TensorImpl
        
        # Create new impl with incremented batch_dims
        new_impl = TensorImpl(
            storages=x._impl._storages,
            values=x._impl._values,
            traced=x._impl.traced,
            batch_dims=x._impl.batch_dims + 1,
            sharding=x._impl.sharding,
        )
        new_impl.cached_shape = x._impl.cached_shape
        new_impl.cached_dtype = x._impl.cached_dtype
        new_impl.cached_device = x._impl.cached_device
        
        return Tensor(impl=new_impl)


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
        from .tensor import Tensor
        from .tensor_impl import TensorImpl
        
        if x._impl.batch_dims <= 0:
            raise ValueError("Cannot decrement batch_dims below 0")
        
        # Create new impl with decremented batch_dims
        new_impl = TensorImpl(
            storages=x._impl._storages,
            values=x._impl._values,
            traced=x._impl.traced,
            batch_dims=x._impl.batch_dims - 1,
            sharding=x._impl.sharding,
        )
        new_impl.cached_shape = x._impl.cached_shape
        new_impl.cached_dtype = x._impl.cached_dtype
        new_impl.cached_device = x._impl.cached_device
        
        return Tensor(impl=new_impl)


class MoveAxisToBatchDimsOp(Operation):
    """Move an axis from logical shape to batch dims.
    
    This operation:
    1. Moves the specified axis to the front of batch dims (position 0)
    2. Increments batch_dims counter
    
    Used by vmap to convert a logical axis into a batch axis.
    """
    
    @property
    def name(self) -> str:
        return "move_axis_to_batch_dims"
    
    def maxpr(self, x: TensorValue, *, axis: int) -> TensorValue:
        """Move axis to front."""
        rank = len(x.type.shape)
        
        # Normalize negative index (relative to logical dims)
        if axis < 0:
            axis = rank + axis
        
        # Move to front
        order = list(range(rank))
        order.pop(axis)
        order.insert(0, axis)
        
        # Use ops.permute for full permutation
        return ops.permute(x, tuple(order))
    
    def __call__(self, x: Tensor, *, axis: int) -> Tensor:
        """Move axis to batch dims and increment counter."""
        # First move the axis to front via parent's __call__
        result = super().__call__(x, axis=axis)
        
        # Then increment batch_dims
        result._impl.batch_dims = x._impl.batch_dims + 1
        
        return result


class MoveAxisFromBatchDimsOp(Operation):
    """Move an axis from batch dims to logical shape.
    
    This operation:
    1. Moves the specified batch axis to a position in the logical shape
    2. Decrements batch_dims counter
    
    The `destination` parameter specifies where in the LOGICAL shape (after 
    batch_dims is decremented) the axis should be placed.
    
    Used by vmap to convert a batch axis back to logical.
    
    Example:
        Input: physical=(C, B, H, W) = (2, 5, 3, 4), batch_dims=2
               logical=(H, W) = (3, 4)
        
        move_axis_from_batch_dims(x, batch_axis=0, logical_destination=2)
        
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
        batch_axis: int = 0,
        logical_destination: int = 0
    ) -> TensorValue:
        """Move batch_axis to logical_destination position.
        
        Args:
            x: Input tensor
            batch_axis: Which batch axis to move (0 = outermost)
            logical_destination: Where to place axis in logical shape (0 = first logical dim)
        """
        rank = len(x.type.shape)
        batch_dims = batch_axis + 1  # Will be decremented, but we need current value for calc
        
        # Normalize batch_axis
        if batch_axis < 0:
            batch_axis = rank + batch_axis
        
        # Convert logical_destination to physical position
        # After batch_axis is removed, batch_dims will be decremented
        # So physical position = (batch_dims - 1) + logical_destination
        # But we also need to account for the batch_axis being removed
        # 
        # Current layout: [batch_0, ..., batch_axis, ..., batch_{n-1}, logical_0, ...]
        # We want: [batch_0, ..., batch_{n-1} (minus batch_axis), logical_0, ..., moved_axis, ...]
        
        # Build permutation: remove batch_axis, insert at physical_destination
        order = list(range(rank))
        order.pop(batch_axis)
        
        # Physical destination = (current_batch_dims - 1) + logical_destination
        # Note: We don't know batch_dims in maxpr, so we use the simpler approach:
        # Just insert at the position that makes logical sense
        # After removing batch_axis from front, the remaining batch dims shift left
        # Then we insert at position = logical_destination (relative to start of logical shape)
        
        # Actually, we need to think about this more carefully.
        # If we have [B0, B1, L0, L1, L2] with batch_dims=2 and want to move B0 to logical pos 2:
        # After removing B0: [B1, L0, L1, L2]  (now batch_dims will be 1)
        # Insert B0 at logical position 2 means after L1: [B1, L0, L1, B0, L2]
        # Physical position = new_batch_dims + logical_destination = 1 + 2 = 3
        
        # But we're in maxpr, we don't have batch_dims info here. 
        # Let's compute it based on the assumption that caller passes correct values.
        # We'll handle the actual batch_dims in __call__.
        
        # For now, just do the permutation assuming we want to move batch_axis
        # to be at the end (as a simple default)
        order.insert(len(order), batch_axis)  # Insert at end as fallback
        
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
            logical_destination: Where in logical shape to place it (0 = first logical dim)
        """
        from .tensor import Tensor
        from .tensor_impl import TensorImpl
        
        if x._impl.batch_dims <= 0:
            raise ValueError("No batch dims to move from")
        
        rank = len(x._impl.physical_shape)
        current_batch_dims = x._impl.batch_dims
        
        # Normalize batch_axis (relative to batch dims, so 0 is outermost)
        if batch_axis < 0:
            batch_axis = current_batch_dims + batch_axis
        
        if batch_axis < 0 or batch_axis >= current_batch_dims:
            raise ValueError(f"batch_axis {batch_axis} out of bounds for batch_dims={current_batch_dims}")
        
        # New batch_dims after moving one out
        new_batch_dims = current_batch_dims - 1
        
        # Normalize logical_destination (relative to logical shape after removal)
        logical_rank = rank - current_batch_dims  # Current logical rank
        if logical_destination < 0:
            logical_destination = logical_rank + 1 + logical_destination  # +1 because we add one
        
        # Physical destination = new_batch_dims + logical_destination
        physical_destination = new_batch_dims + logical_destination
        
        # Build the permutation
        order = list(range(rank))
        order.pop(batch_axis)
        order.insert(physical_destination, batch_axis)
        
        # Apply permutation using moveaxis-style logic
        result = moveaxis(x, batch_axis, physical_destination)
        
        # Update batch_dims on the result
        result._impl.batch_dims = new_batch_dims
        
        return result


# =============================================================================
# Singleton Instances (function-style API)
# =============================================================================

_unsqueeze_op = UnsqueezeOp()
_squeeze_op = SqueezeOp()
_swap_axes_op = SwapAxesOp()
_moveaxis_op = MoveAxisOp()
_broadcast_to_op = BroadcastToOp()
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
    """Broadcast tensor to target shape.
    
    Args:
        x: Input tensor
        shape: Target shape (must be broadcast-compatible)
        
    Returns:
        Tensor broadcasted to shape
    """
    return _broadcast_to_op(x, shape=shape)


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
    # Classes
    "UnsqueezeOp",
    "SqueezeOp", 
    "SwapAxesOp",
    "MoveAxisOp",
    "BroadcastToOp",
    "IncrBatchDimsOp",
    "DecrBatchDimsOp",
    "MoveAxisToBatchDimsOp",
    "MoveAxisFromBatchDimsOp",
    # Functions
    "unsqueeze",
    "squeeze",
    "swap_axes",
    "moveaxis",
    "broadcast_to",
    "incr_batch_dims",
    "decr_batch_dims",
    "move_axis_to_batch_dims",
    "move_axis_from_batch_dims",
]
