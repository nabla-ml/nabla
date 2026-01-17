# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Logical View Operations
# ===----------------------------------------------------------------------=== #

"""Logical view operations.

These operations work on the LOGICAL shape (user's view).
Integer arguments are translated by adding batch_dims offset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops

from .operation import Operation, LogicalAxisOperation, LogicalShapeOperation

if TYPE_CHECKING:
    from ..core import Tensor


class UnsqueezeOp(LogicalAxisOperation):
    axis_offset_for_insert = True
    
    @property
    def name(self) -> str:
        return "unsqueeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Unsqueeze: insert new dimension at axis position."""
        from ..sharding.propagation import OpShardingRuleTemplate
        in_rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        # Input: d0, d1, ...
        factors = [f"d{i}" for i in range(in_rank)]
        in_str = " ".join(factors)
        
        # Output: insert "new_dim" at axis
        out_factors = list(factors)
        out_factors.insert(axis, "new_dim")
        out_str = " ".join(out_factors)
                 
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)
    
    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0]) + 1


class SqueezeOp(LogicalAxisOperation):
    axis_offset_for_insert = False
    
    @property
    def name(self) -> str:
        return "squeeze"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Squeeze: remove dimension at axis position."""
        from ..sharding.propagation import OpShardingRuleTemplate
        in_rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        # Input: d0, ...
        factors = [f"d{i}" for i in range(in_rank)]
        in_str = " ".join(factors)
        
        # Output: remove factor at axis
        out_factors = list(factors)
        out_factors.pop(axis)
        out_str = " ".join(out_factors)
        
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)
    
    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0]) - 1


class SwapAxesOp(LogicalAxisOperation):
    axis_arg_names = ("axis1", "axis2")
    
    @property
    def name(self) -> str:
        return "swap_axes"
    
    def maxpr(self, x: TensorValue, *, axis1: int, axis2: int) -> TensorValue:
        return ops.transpose(x, axis1, axis2)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """SwapAxes: swap two dimensions."""
        from ..sharding.propagation import OpShardingRuleTemplate
        in_rank = len(input_shapes[0])
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        
        factors = [f"d{i}" for i in range(in_rank)]
        in_str = " ".join(factors)
        
        # Output: swap factors
        out_factors = list(factors)
        out_factors[axis1], out_factors[axis2] = out_factors[axis2], out_factors[axis1]
        out_str = " ".join(out_factors)
        
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)
    
    def infer_output_shape(self, input_shapes: list[tuple[int, ...]], **kwargs) -> tuple[int, ...]:
        """Swap dimensions at axis1 and axis2."""
        in_shape = list(input_shapes[0])
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        if axis1 < 0:
            axis1 = len(in_shape) + axis1
        if axis2 < 0:
            axis2 = len(in_shape) + axis2
        in_shape[axis1], in_shape[axis2] = in_shape[axis2], in_shape[axis1]
        return tuple(in_shape)


class BroadcastToOp(LogicalShapeOperation):
    @property
    def name(self) -> str:
        return "broadcast_to"
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        # Simple same-rank broadcast. Any unsqueezing should happen at the Tensor level
        # (in the broadcast_to function) so sharding propagation can handle it.
        return ops.broadcast_to(x, shape)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Broadcast: input dims align to output SUFFIX (numpy semantics).
        
        Uses shape-aware template to handle dimension expansion (size 1 -> N).
        Handles scalar inputs (rank 0) by treating them as fully replicated.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        
        if not input_shapes:
            return None
            
        in_shape = input_shapes[0]
        
        # Use target shape from kwargs if output_shapes not provided (SPMD case)
        out_shape = kwargs.get("shape")
        if out_shape is None:
            if output_shapes:
                out_shape = output_shapes[0]
            else:
                # Should not happen for broadcast_to unless shape arg missing
                return None

        in_rank = len(in_shape)
        out_rank = len(out_shape)
        
        # Handle scalar input (rank 0): no input dims to map
        # The output is purely new dimensions - no sharding propagates from input
        if in_rank == 0:
            out_factors = [f"d{i}" for i in range(out_rank)]
            out_mapping = {i: [out_factors[i]] for i in range(out_rank)}
            # Empty input mapping for scalar
            in_mapping = {}
            return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
                input_shapes, output_shapes
            )
        
        # Factors for output dimensions
        out_factors = [f"d{i}" for i in range(out_rank)]
        out_mapping = {i: [out_factors[i]] for i in range(out_rank)}
        
        # Map input dims to output factors (right-aligned)
        in_mapping = {}
        offset = out_rank - in_rank
        
        for i in range(in_rank):
            # Input dim i corresponds to output dim i + offset
            out_idx = i + offset
            if out_idx >= 0:
                if in_shape[i] == out_shape[out_idx]:
                    # Match: share factor
                    in_mapping[i] = [out_factors[out_idx]]
                else:
                    # Broadcast (1->N) or mismatch: distinct factor
                    in_mapping[i] = [f"bcast_{i}"]
            else:
                 in_mapping[i] = [f"d_extra_{i}"]

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(input_shapes, output_shapes)
    
    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(kwargs.get("shape", input_shapes[0]))

    def _transform_shard_kwargs(self, kwargs: dict, output_sharding, shard_idx: int) -> dict:
        """Convert global target shape to local shape for each shard."""
        from ..sharding.spec import compute_local_shape
        
        global_shape = kwargs.get('shape')
        if global_shape is None or output_sharding is None:
            return kwargs
        
        local_shape = compute_local_shape(global_shape, output_sharding, device_id=shard_idx)
        return {**kwargs, 'shape': local_shape}


class ReshapeOp(LogicalShapeOperation):
    @property
    def name(self) -> str:
        return "reshape"
    
    def __call__(self, x: "Tensor", *, shape: tuple[int, ...]) -> "Tensor":
        """Reshape with conservative sharding safety.
        
        If ANY logical dimension is sharded, we gather the entire tensor
        before reshaping. This is conservative but always correct.
        
        A smarter implementation could analyze the reshape to determine
        if sharded dimensions are actually affected.
        """
        from ..sharding.spec import DimSpec, ShardingSpec
        
        spec = x._impl.sharding
        if spec:
            batch_dims = x._impl.batch_dims
            logical_rank = len(x.shape)
            
            # Check if any logical dimension is sharded
            has_sharded_logical = False
            for i in range(logical_rank):
                phys_idx = batch_dims + i
                if phys_idx < len(spec.dim_specs) and spec.dim_specs[phys_idx].axes:
                    has_sharded_logical = True
                    break
            
            if has_sharded_logical:
                # Conservative: unshard all logical dimensions
                # Keep batch dimensions as-is
                new_dim_specs = []
                for i in range(len(spec.dim_specs)):
                    if i < batch_dims:
                        # Preserve batch dim sharding
                        new_dim_specs.append(spec.dim_specs[i].clone())
                    else:
                        # Unshard logical dims
                        new_dim_specs.append(DimSpec([]))
                
                x = x.with_sharding(spec.mesh, new_dim_specs)
        
        return super().__call__(x, shape=shape)
    
    def maxpr(self, x: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
        return ops.reshape(x, shape)
    
    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(kwargs.get("shape"))

    def sharding_rule(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]], **kwargs):
        """Create sharding rule for reshape using greedy factor matching."""
        from ..sharding.propagation import OpShardingRuleTemplate
        from math import prod
        
        if not input_shapes: return None
        in_shape = input_shapes[0]
        out_shape = kwargs.get('shape')
        if out_shape is None: 
            if output_shapes: out_shape = output_shapes[0]
            else: return None
            
        in_rank = len(in_shape)
        out_rank = len(out_shape)
        
        # Heuristic: The tensor with MORE dimensions has the Atomic Factors
        if in_rank >= out_rank:
            # Input is atomic, Output is compound
            factors = [f"d{i}" for i in range(in_rank)]
            in_mapping = {i: [factors[i]] for i in range(in_rank)}
            out_mapping = {}
            
            factor_idx = 0
            current_prod = 1
            current_factors = []
            
            # Match input factors to output dimensions
            for out_dim_idx in range(out_rank):
                target_size = out_shape[out_dim_idx]
                
                # Consume input factors
                while factor_idx < in_rank:
                    f_size = in_shape[factor_idx]
                    current_factors.append(factors[factor_idx])
                    current_prod *= f_size
                    factor_idx += 1
                    
                    if current_prod == target_size:
                        out_mapping[out_dim_idx] = list(current_factors)
                        current_factors = []
                        current_prod = 1
                        break
                    elif current_prod > target_size:
                        # Split required (not supported by simple propagation yet) - assign all
                        out_mapping[out_dim_idx] = list(current_factors)
                        current_factors = []
                        current_prod = 1
                        break
            
            # Flush remaining
            if list(out_mapping.keys())[-1] != out_rank - 1 and current_factors:
                 # Assign to last dim?
                 pass
                 
        else:
            # Output is atomic, Input is compound
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
                    
                    if current_prod == target_size:
                        in_mapping[in_dim_idx] = list(current_factors)
                        current_factors = []
                        current_prod = 1
                        break
                    elif current_prod > target_size:
                        in_mapping[in_dim_idx] = list(current_factors)
                        current_factors = []
                        current_prod = 1
                        break
                        
        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(input_shapes, output_shapes)
    
    def _transform_shard_kwargs(self, kwargs: dict, output_sharding, shard_idx: int) -> dict:
        """Convert global target shape to local shape for each shard."""
        from ..sharding.spec import compute_local_shape
        
        global_shape = kwargs.get('shape')
        if global_shape is None or output_sharding is None:
            return kwargs
        
        local_shape = compute_local_shape(global_shape, output_sharding, device_id=shard_idx)
        return {**kwargs, 'shape': local_shape}


_unsqueeze_op = UnsqueezeOp()
_squeeze_op = SqueezeOp()
_swap_axes_op = SwapAxesOp()
_broadcast_to_op = BroadcastToOp()
_reshape_op = ReshapeOp()


class SliceTensorOp(Operation):
    @property
    def name(self) -> str:
        return "slice_tensor"
    
    def maxpr(self, x: TensorValue, start: Any, size: Any) -> TensorValue:
        # data is TensorValue
        # start/size are lists of int or TensorValue
        slices = []
        for s, sz in zip(start, size):
            end = s + sz
            slices.append(slice(s, end))
        return ops.slice_tensor(x, slices)

    # sharding_rule inherited from Operation (elementwise identity)
        
    def infer_output_rank(self, input_shapes, **kwargs) -> int:
        return len(input_shapes[0])

_slice_tensor_op = SliceTensorOp()


def unsqueeze(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_op(x, axis=axis)

def squeeze(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_op(x, axis=axis)

def swap_axes(x: Tensor, axis1: int, axis2: int) -> Tensor:
    return _swap_axes_op(x, axis1=axis1, axis2=axis2)

def broadcast_to(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    # Numpy broadcast aligns from the right. If input has fewer dims than target,
    # we need to unsqueeze to add the missing dimensions.
    #
    # For batched tensors: physical shape = (batch..., logical...)
    # We insert the new dims AFTER the batch dims so broadcast aligns correctly.
    # E.g., scalar with batch_dims=1, logical shape () -> broadcast to (8,)
    #       physical (4,) -> unsqueeze at axis=1 -> (4, 1) -> broadcast to (4, 8)
    #
    # IMPORTANT: This must happen at Tensor level (not in maxpr) so that
    # unsqueeze operations go through sharding propagation.
    in_logical_rank = len(x.shape)
    out_logical_rank = len(shape)
    
    if in_logical_rank < out_logical_rank:
        batch_dims = x._impl.batch_dims
        # Insert dimensions after batch dims (at logical position 0, which is physical batch_dims)
        for _ in range(out_logical_rank - in_logical_rank):
            x = unsqueeze(x, axis=0)  # Logical axis 0 -> physical axis batch_dims
    
    return _broadcast_to_op(x, shape=shape)

def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _reshape_op(x, shape=shape)

def slice_tensor(x: Tensor, start: Any, size: Any) -> Tensor:
    return _slice_tensor_op(x, start=start, size=size)


# =============================================================================
# Gather Operation - Properly Implemented
# =============================================================================

class GatherOp(Operation):
    """Gather elements from data tensor along an axis using indices.
    
    For axis=0: data[i0, i1, ...] with indices[k0, k1, ...] gives
    output[k0, k1, ..., i1, ...] where the axis dimension is replaced
    by the indices dimensions.
    
    Uses Operation base class (not LogicalAxisOperation) because gather
    takes multiple tensor inputs and needs custom axis handling.
    """
    
    @property
    def name(self) -> str:
        return "gather"
    
    def __call__(self, x: "Tensor", indices: "Tensor", *, axis: int = 0) -> "Tensor":
        """Call gather with logical axis translation."""
        data_batch_dims = x._impl.batch_dims
        indices_batch_dims = indices._impl.batch_dims
        logical_ndim = len(x.shape)
        
        # Translate logical axis to physical axis
        if axis < 0:
            axis = logical_ndim + axis
        phys_axis = data_batch_dims + axis
        
        # Pass batch_dims info to maxpr for proper batched gather handling
        # Only use batched gather when BOTH inputs have matching batch dims
        use_batched = (data_batch_dims > 0 and indices_batch_dims > 0 
                       and data_batch_dims == indices_batch_dims)
        batch_dims = data_batch_dims if use_batched else 0
        
        return super().__call__(x, indices, axis=phys_axis, batch_dims=batch_dims)
    
    def maxpr(self, x: TensorValue, indices: TensorValue, *, axis: int = 0, batch_dims: int = 0) -> TensorValue:
        from max.dtype import DType
        
        if batch_dims > 0:
            # Use gather_nd with batch_dims for batched gather
            # gather_nd expects indices of shape (..., index_depth) where last dim indexes into data
            # For 1D gather along a logical axis, we reshape indices to add a trailing dim
            indices_shape = list(indices.shape)
            
            # Reshape indices: (..., k) -> (..., k, 1) for 1D indexing
            new_shape = indices_shape + [1]
            indices = ops.reshape(indices, new_shape)
            
            # Cast to int64 if needed
            if indices.dtype != DType.int64:
                indices = ops.cast(indices, DType.int64)
            
            return ops.gather_nd(x, indices, batch_dims=batch_dims)
        else:
            # Simple case: use regular gather
            return ops.gather(x, indices, axis)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Gather sharding rule: Data(d...), Indices(i...) 
        -> Output(d_prefix..., i..., d_suffix...).
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        
        if not input_shapes or len(input_shapes) < 2:
            return None
        
        data_rank = len(input_shapes[0])
        indices_rank = len(input_shapes[1])
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis += data_rank
            
        # Data factors: d0, ...
        data_factors = [f"d{i}" for i in range(data_rank)]
        data_str = " ".join(data_factors)
        
        # Indices factors: i0, ...
        indices_factors = [f"i{i}" for i in range(indices_rank)]
        indices_str = " ".join(indices_factors)
        
        # Output factors: replace axis factor with indices factors
        out_factors = data_factors[:axis] + indices_factors + data_factors[axis+1:]
        out_str = " ".join(out_factors)
        
        return OpShardingRuleTemplate.parse(f"{data_str}, {indices_str} -> {out_str}", input_shapes).instantiate(
            input_shapes, output_shapes
        )


_gather_op = GatherOp()


def gather(x: "Tensor", indices: "Tensor", axis: int = 0) -> "Tensor":
    """Gather elements from x along axis using indices.
    
    Args:
        x: Data tensor to gather from
        indices: Index tensor (integer indices)
        axis: Axis along which to gather (default 0)
        
    Returns:
        Gathered tensor where axis dimension is replaced by indices dimensions
    """
    from .operation import ensure_tensor
    indices = ensure_tensor(indices)
    return _gather_op(x, indices, axis=axis)


# =============================================================================
# Scatter Operation - Properly Implemented
# =============================================================================

class ScatterOp(Operation):
    """Scatter updates into data tensor at indices along an axis.
    
    For axis=0: data[indices[k], ...] = updates[k, ...]
    Output has same shape as input data.
    
    Uses Operation base class (not LogicalAxisOperation) because scatter
    takes multiple tensor inputs and needs custom axis handling.
    """
    
    @property
    def name(self) -> str:
        return "scatter"
    
    def __call__(
        self, 
        x: "Tensor", 
        indices: "Tensor", 
        updates: "Tensor", 
        *, 
        axis: int = 0
    ) -> "Tensor":
        """Call scatter with logical axis translation."""
        batch_dims = x._impl.batch_dims
        logical_ndim = len(x.shape)
        
        # Translate logical axis to physical axis
        if axis < 0:
            axis = logical_ndim + axis
        phys_axis = batch_dims + axis
        
        return super().__call__(x, indices, updates, axis=phys_axis)
    
    def maxpr(
        self, 
        x: TensorValue, 
        indices: TensorValue, 
        updates: TensorValue, 
        *, 
        axis: int = 0
    ) -> TensorValue:
        from max.dtype import DType
        
        # Use scatter_nd which has more flexible index handling.
        # scatter_nd expects indices of shape (..., index_depth) where index_depth
        # is the number of dimensions to index into.
        #
        # For scatter along axis=0 into (8, 4) with 1D indices (2,) and updates (2, 4):
        # - We reshape indices from (2,) to (2, 1) to indicate 1-dimension indexing
        # - updates shape (2, 4) provides the values for each index
        #
        # For scatter along axis=1 into (4, 8) with 1D indices (3,) and updates (4, 3):
        # - We need to expand indices to include all row indices
        # - indices becomes shape (4, 3, 2) - for each (row, update_idx), give (row, col)
        
        indices_shape = list(indices.shape)
        
        if axis == 0:
            # Simple case: just add trailing dimension to indices
            # indices (k,) -> (k, 1) for k updates along first axis
            new_indices_shape = indices_shape + [1]
            indices = ops.reshape(indices, new_indices_shape)
            # Cast to int64 if needed (scatter_nd expects int64)
            if indices.dtype != DType.int64:
                indices = ops.cast(indices, DType.int64)
            return ops.scatter_nd(x, updates, indices)
        else:
            # For axis != 0, we need to construct full coordinate indices
            # For axis=1, data (4, 8), indices (3,), updates (4, 3)
            # We need indices of shape (4, 3, 2) where each (i, j, :) = (i, indices[j])
            
            leading_dims = [int(d) for d in x.shape[:axis]]
            trailing_update_dims = [int(d) for d in indices_shape]
            full_shape = leading_dims + trailing_update_dims
            
            coord_list = []
            
            # Add leading dimension coordinates
            for d, dim_size in enumerate(leading_dims):
                # Create range for this dimension using ops.range
                from max.graph import DeviceRef
                coord = ops.range(0, dim_size, 1, dtype=DType.int64, device=DeviceRef.CPU())
                # Reshape for broadcasting: insert 1s everywhere except position d
                shape = [1] * len(full_shape)
                shape[d] = dim_size
                coord = ops.reshape(coord, shape)
                # Broadcast to full_shape
                coord = ops.broadcast_to(coord, full_shape)
                coord_list.append(coord)
            
            # Add the actual indices (for the scatter axis)
            idx = indices
            if idx.dtype != DType.int64:
                idx = ops.cast(idx, DType.int64)
            # Reshape: add leading 1s for broadcasting
            shape = [1] * len(leading_dims) + trailing_update_dims
            idx = ops.reshape(idx, shape)
            idx = ops.broadcast_to(idx, full_shape)
            coord_list.append(idx)
            
            # Stack coordinates along last axis
            stacked = ops.stack(coord_list, axis=-1)
            
            return ops.scatter_nd(x, updates, stacked)
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Scatter sharding rule: Data(d...), Indices(i...), Updates(i..., d_suffix...) -> Data.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        
        if not input_shapes or len(input_shapes) < 3:
            return None
        
        data_rank = len(input_shapes[0])
        indices_rank = len(input_shapes[1])
        axis = kwargs.get("axis", 0)
        if axis < 0:
            axis += data_rank
            
        # Data factors
        data_factors = [f"d{i}" for i in range(data_rank)]
        data_str = " ".join(data_factors)
        
        # Indices factors
        indices_factors = [f"i{i}" for i in range(indices_rank)]
        indices_str = " ".join(indices_factors)
        
        # Updates factors: match gather output
        updates_factors = data_factors[:axis] + indices_factors + data_factors[axis+1:]
        updates_str = " ".join(updates_factors)

        # Output maps back to data (d0 ...)
        out_str = data_str
        
        return OpShardingRuleTemplate.parse(f"{data_str}, {indices_str}, {updates_str} -> {out_str}", input_shapes).instantiate(
            input_shapes, output_shapes
        )


_scatter_op = ScatterOp()


def scatter(x: "Tensor", indices: "Tensor", updates: "Tensor", axis: int = 0) -> "Tensor":
    """Scatter updates into x at indices along axis.
    
    Args:
        x: Data tensor to scatter into
        indices: Index tensor (integer indices)
        updates: Values to scatter (shape: indices_shape + data_shape[axis+1:])
        axis: Axis along which to scatter (default 0)
        
    Returns:
        Updated tensor with same shape as x
    """
    from .operation import ensure_tensor
    x = ensure_tensor(x)
    indices = ensure_tensor(indices)
    updates = ensure_tensor(updates)
    return _scatter_op(x, indices, updates, axis=axis)


# =============================================================================
# Concatenate Operation - Properly Implemented
# =============================================================================

class ConcatenateOp(LogicalAxisOperation):
    """Concatenate tensors along an axis.
    
    All input tensors must have same shape except along concat axis.
    The concat axis dimension is summed.
    
    Following the established pattern from LogicalAxisOperation.
    """
    
    @property
    def name(self) -> str:
        return "concatenate"
    
    def __call__(self, tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        """Override to handle list of tensors with batch_dims adjustment."""
        if not tensors:
            raise ValueError("concatenate expects at least one tensor")
        
        first = tensors[0]
        batch_dims = first._impl.batch_dims
        
        # Adjust axis for batch_dims (logical -> physical)
        if axis < 0:
            axis += len(first.shape)
        phys_axis = batch_dims + axis
        
        # Call parent Operation.__call__ directly (skip LogicalAxisOperation)
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
        
        CRITICAL FIX: Concat axis uses a SHARED factor 'c_concat' across all inputs and output.
        This effectively treats the concat axis as "elastic" but requires uniform sharding.
        If inputs are sharded on 'x', output must be sharded on 'x'.
        The fact that input sizes sum to output size is handled by runtime, not propagation factors.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        
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
                    # Shared factor for concat axis -> enforces sharding consistency
                    mapping[dim] = ["c_concat"]
                else:
                    mapping[dim] = [f"d{dim}"]
            input_mappings.append(mapping)
            
        # Output mapping
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
        """Output has same rank as inputs."""
        return len(input_shapes[0])


_concatenate_op = ConcatenateOp()


def concatenate(tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
    """Concatenate tensors along an axis.
    
    Args:
        tensors: Sequence of tensors to concatenate
        axis: Axis along which to concatenate (default 0)
        
    Returns:
        Concatenated tensor
    """
    return _concatenate_op(tensors, axis=axis)


def stack(tensors: list["Tensor"], axis: int = 0) -> "Tensor":
    """Stack tensors along a new axis.
    
    All tensors must have the same shape. A new dimension is inserted
    at axis position.
    
    Args:
        tensors: List of tensors to stack
        axis: Position of new axis (default 0)
        
    Returns:
        Stacked tensor with one more dimension than inputs
    """
    if not tensors:
        raise ValueError("stack requires at least one tensor")
    
    # 1. Unsqueeze all inputs at the stack axis
    expanded = [unsqueeze(t, axis=axis) for t in tensors]
    
    # 2. Concatenate along that axis
    return concatenate(expanded, axis=axis)


__all__ = [
    "UnsqueezeOp", "SqueezeOp", "SwapAxesOp", "BroadcastToOp", "ReshapeOp", "SliceTensorOp",
    "GatherOp", "ScatterOp", "ConcatenateOp",
    "unsqueeze", "squeeze", "swap_axes", "broadcast_to", "reshape", "slice_tensor",
    "gather", "scatter", "concatenate", "stack",
]


