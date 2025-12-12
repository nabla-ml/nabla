# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Multi-output operations for the eager module.

These operations demonstrate the pytree-based multi-output support,
returning tuples, lists, or dicts of Tensors.
"""

from __future__ import annotations

from typing import Any

from max.graph import TensorValue, ops

from .ops import Operation


class SplitOp(Operation):
    """Split a tensor into multiple equal chunks along an axis.
    
    Example:
        >>> x = Tensor.arange(0, 12).reshape((3, 4))
        >>> a, b = split(x, num_splits=2, axis=1)
        >>> # a.shape = (3, 2), b.shape = (3, 2)
    """
    
    @property
    def name(self) -> str:
        return "split"
    
    def maxpr(
        self, 
        x: TensorValue, 
        *, 
        num_splits: int, 
        axis: int = 0
    ) -> tuple[TensorValue, ...]:
        """Split tensor into num_splits equal parts along axis.
        
        Args:
            x: Input TensorValue
            num_splits: Number of equal splits
            axis: Axis to split along (default: 0)
            
        Returns:
            Tuple of TensorValues, one for each split
        """
        # Get axis size and compute chunk sizes
        shape = list(x.type.shape)
        axis_size = int(shape[axis])
        
        if axis_size % num_splits != 0:
            raise ValueError(
                f"Cannot split axis of size {axis_size} into {num_splits} equal parts"
            )
        
        chunk_size = axis_size // num_splits
        split_sizes = [chunk_size] * num_splits
        
        # Use MAX's native split which returns list[TensorValue]
        result_list = ops.split(x, split_sizes, axis)
        return tuple(result_list)


class ChunkOp(Operation):
    """Split a tensor into a specified number of chunks.
    
    Similar to SplitOp but returns a list instead of tuple.
    
    Example:
        >>> x = Tensor.arange(0, 12)
        >>> chunks = chunk(x, chunks=3)
        >>> # chunks is a list of 3 Tensors
    """
    
    @property
    def name(self) -> str:
        return "chunk"
    
    def maxpr(
        self, 
        x: TensorValue, 
        *, 
        chunks: int, 
        axis: int = 0
    ) -> list[TensorValue]:
        """Split tensor into specified number of chunks.
        
        Args:
            x: Input TensorValue
            chunks: Number of chunks
            axis: Axis to split along (default: 0)
            
        Returns:
            List of TensorValues
        """
        shape = list(x.type.shape)
        axis_size = int(shape[axis])
        chunk_size = (axis_size + chunks - 1) // chunks  # ceiling division
        
        # Compute chunk sizes (last chunk may be smaller)
        split_sizes = []
        remaining = axis_size
        for i in range(chunks):
            if remaining <= 0:
                break
            size = min(chunk_size, remaining)
            split_sizes.append(size)
            remaining -= size
        
        # Use MAX's native split
        return ops.split(x, split_sizes, axis)


class UnbindOp(Operation):
    """Remove a dimension and return tuple of slices.
    
    Example:
        >>> x = Tensor.zeros((3, 4, 5))
        >>> slices = unbind(x, axis=0)
        >>> # slices is a tuple of 3 Tensors, each with shape (4, 5)
    """
    
    @property
    def name(self) -> str:
        return "unbind"
    
    def maxpr(
        self, 
        x: TensorValue, 
        *, 
        axis: int = 0
    ) -> tuple[TensorValue, ...]:
        """Remove dimension and return slices.
        
        Args:
            x: Input TensorValue
            axis: Axis to unbind (default: 0)
            
        Returns:
            Tuple of TensorValues with dim removed
        """
        shape = list(x.type.shape)
        axis_size = int(shape[axis])
        
        # Split into individual slices of size 1
        split_sizes = [1] * axis_size
        sliced = ops.split(x, split_sizes, axis)
        
        # Squeeze out the axis dimension from each slice
        results = [ops.squeeze(s, axis) for s in sliced]
        return tuple(results)


class MinMaxOp(Operation):
    """Return both min and max of a tensor (example of dict output).
    
    Example:
        >>> x = Tensor.arange(0, 10)
        >>> result = minmax(x)
        >>> result['min'], result['max']
    """
    
    @property
    def name(self) -> str:
        return "minmax"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> dict[str, TensorValue]:
        """Compute min and max simultaneously.
        
        Args:
            x: Input TensorValue
            
        Returns:
            Dict with 'min' and 'max' keys
        """
        return {
            'min': ops.min(x),
            'max': ops.max(x),
        }


# ===== Singleton instances exposed as functions =====

split = SplitOp()
chunk = ChunkOp()
unbind = UnbindOp()
minmax = MinMaxOp()


__all__ = [
    "SplitOp",
    "ChunkOp",
    "UnbindOp",
    "MinMaxOp",
    "split",
    "chunk",
    "unbind",
    "minmax",
]
