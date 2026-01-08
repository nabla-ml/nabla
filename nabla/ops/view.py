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

from .operation import LogicalAxisOperation, LogicalShapeOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


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
        from ..sharding.propagation import unsqueeze_template
        in_rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        return unsqueeze_template(in_rank, axis).instantiate(input_shapes, output_shapes)
    
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
        from ..sharding.propagation import squeeze_template
        in_rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        return squeeze_template(in_rank, axis).instantiate(input_shapes, output_shapes)
    
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
        """SwapAxes: swap two dimensions and their shardings."""
        from ..sharding.propagation import swap_axes_template
        in_rank = len(input_shapes[0])
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        return swap_axes_template(in_rank, axis1, axis2).instantiate(input_shapes, output_shapes)
    
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
        """
        from ..sharding.propagation import broadcast_with_shapes_template
        in_shape = input_shapes[0]
        
        # Use target shape from kwargs if output_shapes not provided (SPMD case)
        out_shape = kwargs.get("shape")
        if out_shape is None:
            if output_shapes:
                out_shape = output_shapes[0]
            else:
                # Should not happen for broadcast_to unless shape arg missing
                return None

        return broadcast_with_shapes_template(in_shape, out_shape).instantiate(input_shapes, output_shapes)
    
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
        """Create sharding rule for reshape using compound factors."""
        from ..sharding.propagation import reshape_template
        
        target_shape = kwargs.get('shape')
        if target_shape is None:
            return None
            
        return reshape_template(input_shapes[0], target_shape).instantiate(input_shapes, output_shapes)
    
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


__all__ = [
    "UnsqueezeOp", "SqueezeOp", "SwapAxesOp", "BroadcastToOp", "ReshapeOp",
    "unsqueeze", "squeeze", "swap_axes", "broadcast_to", "reshape",
]
