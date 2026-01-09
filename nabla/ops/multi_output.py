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

"""Multi-output operations for the nabla module.

These operations demonstrate the pytree-based multi-output support,
returning tuples, lists, or dicts of Tensors.
"""

from __future__ import annotations

from typing import Any

from max.graph import TensorValue, ops

from .operation import Operation, LogicalAxisOperation


class SplitOp(LogicalAxisOperation):
    """Split a tensor into multiple equal chunks along an axis.
    
    Example:
        >>> x = Tensor.arange(0, 12).reshape((3, 4))
        >>> a, b = split(x, num_splits=2, axis=1)
        >>> # a.shape = (3, 2), b.shape = (3, 2)
    """
    
    @property
    def name(self) -> str:
        return "split"
    
    def __call__(self, x: Tensor, **kwargs: Any) -> tuple[Tensor, ...]:
        # Enforce replication on split axis to ensure correct global result semantics
        # If we split a sharded axis without gathering, we get partial results 
        # that don't match the global shape expectation of 'Replicated'.
        
        from ..sharding.spec import DimSpec, ShardingSpec
        
        # 1. Determine physical split axis
        rank = len(x.shape)
        batch_dims = x._impl.batch_dims
        axis = kwargs.get("axis", 0)
        
        if axis < 0:
            axis = rank + axis
        
        phys_axis = batch_dims + axis
        
        # 2. Check if input is sharded on this axis
        spec = x._impl.sharding
        if spec and phys_axis < len(spec.dim_specs):
            ds = spec.dim_specs[phys_axis]
            if ds.axes:  # Check if sharded (non-empty axes)
                # 3. Reshard to remove sharding on this axis (AllGather)
                new_dim_specs = list(spec.dim_specs)
                new_dim_specs[phys_axis] = DimSpec([]) # Replicated
                
                x = x.with_sharding(spec.mesh, new_dim_specs)
        
        return super().__call__(x, **kwargs)

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
        
        print(f"DEBUG: SplitOp.maxpr called with num_splits={num_splits}, axis={axis}, axis_size={axis_size}, shape={shape}")

        
        if axis_size % num_splits != 0:
            raise ValueError(
                f"Cannot split axis of size {axis_size} into {num_splits} equal parts"
            )
        
        chunk_size = axis_size // num_splits
        split_sizes = [chunk_size] * num_splits
        
        # Use MAX's native split which returns list[TensorValue]
        result_list = ops.split(x, split_sizes, axis)
        return tuple(result_list)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Split preserves sharding on non-split dimensions.
        
        The split axis changes size, so we must assign a new factor to it
        in the output to avoid size mismatch conflicts.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        if axis < 0: axis += rank
        
        # Input factors: a, b, c...
        in_factors = [chr(97 + i) for i in range(rank)]
        
        # Output factors: change split axis to 'z'
        out_factors = list(in_factors)
        out_factors[axis] = 'z'
        
        in_mapping = {i: [in_factors[i]] for i in range(rank)}
        out_mapping = {i: [out_factors[i]] for i in range(rank)}
        
        if output_shapes is not None:
            count = len(output_shapes)
        else:
            count = kwargs.get("num_splits", 2)
        
        return OpShardingRuleTemplate(
            input_mappings=[in_mapping],
            output_mappings=[out_mapping] * count
        ).instantiate(input_shapes, output_shapes)
    
    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])  # Same rank as input


class ChunkOp(LogicalAxisOperation):
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
    def __call__(self, x: Tensor, **kwargs: Any) -> list[Tensor]:
        # Enforce replication on split axis (see SplitOp.__call__ for details)
        from ..sharding.spec import DimSpec, ShardingSpec
        
        rank = len(x.shape)
        batch_dims = x._impl.batch_dims
        axis = kwargs.get("axis", 0)
        
        if axis < 0:
            axis = rank + axis
        
        phys_axis = batch_dims + axis
        
        spec = x._impl.sharding
        if spec and phys_axis < len(spec.dim_specs):
            ds = spec.dim_specs[phys_axis]
            if ds.axes:  # Check if sharded (non-empty axes)
                # 3. Reshard to remove sharding on this axis (AllGather)
                new_dim_specs = list(spec.dim_specs)
                new_dim_specs[phys_axis] = DimSpec([]) # Replicated
                
                x = x.with_sharding(spec.mesh, new_dim_specs)
        
        return super().__call__(x, **kwargs)

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
        
        # Numpy-style splitting (distribute remainder)
        div = axis_size // chunks
        rem = axis_size % chunks
        
        split_sizes = []
        for i in range(chunks):
            size = div + 1 if i < rem else div
            # Skip empty chunks if checking strict length
            if size > 0:
                split_sizes.append(size)
        
        # Use MAX's native split
        return ops.split(x, split_sizes, axis)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Chunk preserves sharding on non-split dimensions."""
        from ..sharding.propagation import OpShardingRuleTemplate
        
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        if axis < 0: axis += rank
        
        # Input factors: a, b, c...
        in_factors = [chr(97 + i) for i in range(rank)]
        
        # Output factors: change split axis to 'z'
        out_factors = list(in_factors)
        out_factors[axis] = 'z'
        
        in_mapping = {i: [in_factors[i]] for i in range(rank)}
        out_mapping = {i: [out_factors[i]] for i in range(rank)}
        
        if output_shapes is not None:
            count = len(output_shapes)
        else:
            # Calculate expected number of chunks based on logic
            # input_shapes[0] is (d0, d1, ...)
            # sharding_rule might see abstract shapes? No, concrete integers in input_shapes.
            chunks = kwargs.get("chunks", 1)
            dim_size = input_shapes[0][axis]
            
            div = dim_size // chunks
            rem = dim_size % chunks
            count = 0
            for i in range(chunks):
                s = div + 1 if i < rem else div
                if s > 0: count += 1
        
        return OpShardingRuleTemplate(
            input_mappings=[in_mapping],
            output_mappings=[out_mapping] * count
        ).instantiate(input_shapes, output_shapes)
    
    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])  # Same rank as input


class UnbindOp(LogicalAxisOperation):
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
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Unbind: the unbound axis is removed, other dims shift.
        
        Outputs (N slices) all share the same sharding: factor at axis is gone.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        factors = [f"d{i}" for i in range(rank)]
        in_mapping = {i: [factors[i]] for i in range(rank)}
        
        # Remove factor at axis
        out_factors = [factors[i] for i in range(rank) if i != axis]
        out_mapping = {i: [out_factors[i]] for i in range(len(out_factors))}
        
        # Determine number of outputs (N)
        # Unbind returns 'axis_size' outputs.
        # But sharding_rule might be called with concrete input_shapes containing integer dims.
        if output_shapes:
            count = len(output_shapes)
        else:
            count = input_shapes[0][axis]

        return OpShardingRuleTemplate([in_mapping], [out_mapping] * count).instantiate(input_shapes, output_shapes)
    
    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0]) - 1  # One dimension removed


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
