# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from ..base import LogicalAxisOperation, Operation

if TYPE_CHECKING:
    from ...core import Tensor


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
        from ...core.sharding.propagation import OpShardingRuleTemplate
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
        from ...core.sharding.propagation import OpShardingRuleTemplate
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
        from ...core.sharding.propagation import OpShardingRuleTemplate
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


# Singleton instances
_unsqueeze_op = UnsqueezeOp()
_squeeze_op = SqueezeOp()
_swap_axes_op = SwapAxesOp()

__all__ = [
    "unsqueeze", "squeeze", "swap_axes",
    "MoveAxisOp", "UnsqueezePhysicalOp", "SqueezePhysicalOp",
    "moveaxis", "unsqueeze_physical", "squeeze_physical",
]

# Public API wrappers
def unsqueeze(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_op(x, axis=axis)

def squeeze(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_op(x, axis=axis)

def swap_axes(x: Tensor, axis1: int, axis2: int) -> Tensor:
    return _swap_axes_op(x, axis1=axis1, axis2=axis2)

# =============================================================================
# Physical Axis Ops
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
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ...core.sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        source = kwargs.get("source")
        destination = kwargs.get("destination")
        
        # Normalize axes
        if source < 0: source += rank
        if destination < 0: destination += rank
        
        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)
        
        # Calculate permutation
        perm = list(factors)
        val = perm.pop(source)
        perm.insert(destination, val)
        out_str = " ".join(perm)
        
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)


class UnsqueezePhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "unsqueeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.unsqueeze(x, axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ...core.sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)
        
        # Insert "new_dim"
        out_factors = list(factors)
        out_factors.insert(axis, "new_dim")
        out_str = " ".join(out_factors)
                
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)
    
    def infer_output_rank(self, input_shapes: tuple[tuple[int, ...], ...], **kwargs) -> int:
        return len(input_shapes[0]) + 1


class SqueezePhysicalOp(Operation):
    @property
    def name(self) -> str:
        return "squeeze_physical"
    
    def maxpr(self, x: TensorValue, *, axis: int = 0) -> TensorValue:
        return ops.squeeze(x, axis)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ) -> Any:
        from ...core.sharding.propagation import OpShardingRuleTemplate
        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)
        
        factors = [f"d{i}" for i in range(rank)]
        in_str = " ".join(factors)
        
        # Remove factor at axis
        out_factors = list(factors)
        out_factors.pop(axis)
        out_str = " ".join(out_factors)
                
        return OpShardingRuleTemplate.parse(f"{in_str} -> {out_str}", input_shapes).instantiate(input_shapes, output_shapes)

    def infer_output_rank(self, input_shapes: tuple[tuple[int, ...], ...], **kwargs) -> int:
        return len(input_shapes[0]) - 1

_moveaxis_op = MoveAxisOp()
_unsqueeze_physical_op = UnsqueezePhysicalOp()
_squeeze_physical_op = SqueezePhysicalOp()

def moveaxis(x: Tensor, source: int, destination: int) -> Tensor:
    return _moveaxis_op(x, source=source, destination=destination)

def unsqueeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    return _unsqueeze_physical_op(x, axis=axis)

def squeeze_physical(x: Tensor, axis: int = 0) -> Tensor:
    return _squeeze_physical_op(x, axis=axis)
