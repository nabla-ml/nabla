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

"""Reduction operations for the eager module.

These operations reduce tensor dimensions. They are batch_dims-aware:
- The `axis` parameter is interpreted as a LOGICAL axis (user's view)
- Batch dimensions are preserved and not reduced

Example:
    # Physical shape: (batch=2, rows=3, cols=4), batch_dims=1
    # Logical shape: (rows=3, cols=4)
    
    y = reduce_sum(x, axis=0)  # axis=0 reduces ROWS (logical axis)
    
    # Physical output: (batch=2, cols=4), batch_dims=1
    # Logical output: (cols=4)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from .ops import Operation

if TYPE_CHECKING:
    from .tensor import Tensor


# =============================================================================
# Sum Reduction
# =============================================================================

class ReduceSumOp(Operation):
    """Sum reduction over a specified axis.
    
    CRITICAL: axis is a LOGICAL axis (relative to user-visible shape).
    Batch dimensions are preserved.
    """
    
    @property
    def name(self) -> str:
        return "reduce_sum"
    
    def maxpr(
        self, 
        x: TensorValue, 
        *, 
        axis: int,
        keepdims: bool = False,
    ) -> TensorValue:
        """Reduce by sum over axis (receives PHYSICAL axis)."""
        # ops.sum ALWAYS keeps dimensions (reduces to size 1)
        result = ops.sum(x, axis=axis)
        
        if not keepdims:
            # Squeeze if we shouldn't keep dimensions
            result = ops.squeeze(result, axis=axis)
            
        return result
    
    def __call__(
        self, 
        x: Tensor, 
        *, 
        axis: int,
        keepdims: bool = False,
    ) -> Tensor:
        """Sum over LOGICAL axis, preserving batch dimensions.
        
        Args:
            x: Input tensor
            axis: Logical axis to reduce (can be negative)
            keepdims: If True, keep reduced axis as size 1
            
        Returns:
            Tensor with sum over specified axis
        """
        batch_dims = x._impl.batch_dims
        logical_ndim = len(x.shape)
        
        # Normalize negative axis (relative to logical shape)
        if axis < 0:
            axis = logical_ndim + axis
        if axis < 0 or axis >= logical_ndim:
            raise ValueError(
                f"axis {axis} out of bounds for tensor with "
                f"logical shape {tuple(x.shape)}"
            )
        
        # Translate to physical axis
        physical_axis = batch_dims + axis
        
        # Call base class with translated axis
        result = super().__call__(x, axis=physical_axis, keepdims=keepdims)
        
        # Preserve batch_dims on output
        result._impl.batch_dims = batch_dims
        
        return result


_reduce_sum_op = ReduceSumOp()


def reduce_sum(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    """Sum over a logical axis.
    
    The axis is interpreted relative to the LOGICAL shape (user's view),
    not the physical shape. Batch dimensions are preserved.
    
    Args:
        x: Input tensor
        axis: Logical axis to reduce (supports negative indexing)
        keepdims: If True, keep reduced dimension as size 1
        
    Returns:
        Tensor with sum over specified axis
        
    Example:
        # Inside vmap: physical (batch=5, rows=3, cols=4), batch_dims=1
        # User sees logical: (rows=3, cols=4)
        
        y = reduce_sum(x, axis=0)  # Reduces rows
        # Output: logical (cols=4), physical (batch=5, cols=4)
    """
    return _reduce_sum_op(x, axis=axis, keepdims=keepdims)


__all__ = [
    "ReduceSumOp",
    "reduce_sum",
]
