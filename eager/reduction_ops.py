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

All reduction ops inherit from ReduceOperation ABC which handles:
- batch_dims-aware axis translation for vmap support
- The `axis` parameter is interpreted as a LOGICAL axis (user's view)
- Batch dimensions are preserved and not reduced

Subclasses only need to implement:
- name property
- maxpr(x, *, axis, keepdims) -> TensorValue

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

from .ops import ReduceOperation

if TYPE_CHECKING:
    from .tensor import Tensor


# =============================================================================
# Sum Reduction
# =============================================================================

class ReduceSumOp(ReduceOperation):
    """Sum reduction over a specified axis.
    
    The axis is interpreted as a LOGICAL axis. Batch dimensions are preserved.
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


# =============================================================================
# Mean Reduction
# =============================================================================

class MeanOp(ReduceOperation):
    """Mean reduction over a specified axis.
    
    The axis is interpreted as a LOGICAL axis. Batch dimensions are preserved.
    """
    
    @property
    def name(self) -> str:
        return "mean"
    
    def maxpr(
        self, 
        x: TensorValue, 
        *, 
        axis: int,
        keepdims: bool = False,
    ) -> TensorValue:
        """Reduce by mean over axis (receives PHYSICAL axis)."""
        # ops.mean ALWAYS keeps dimensions (reduces to size 1)
        result = ops.mean(x, axis=axis)
        
        if not keepdims:
            # Squeeze if we shouldn't keep dimensions
            result = ops.squeeze(result, axis=axis)
            
        return result


_mean_op = MeanOp()


def mean(x: Tensor, *, axis: int, keepdims: bool = False) -> Tensor:
    """Mean over a logical axis.
    
    The axis is interpreted relative to the LOGICAL shape (user's view),
    not the physical shape. Batch dimensions are preserved.
    
    Args:
        x: Input tensor
        axis: Logical axis to reduce (supports negative indexing)
        keepdims: If True, keep reduced dimension as size 1
        
    Returns:
        Tensor with mean over specified axis
        
    Example:
        # Inside vmap: physical (batch=5, rows=3, cols=4), batch_dims=1
        # User sees logical: (rows=3, cols=4)
        
        y = mean(x, axis=0)  # Mean over rows
        # Output: logical (cols=4), physical (batch=5, cols=4)
    """
    return _mean_op(x, axis=axis, keepdims=keepdims)


__all__ = [
    "ReduceSumOp",
    "reduce_sum",
    "MeanOp",
    "mean",
]

