# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Updated Binary Operations
# ===----------------------------------------------------------------------=== #

"""Binary ops using updated Operation base with auto batch_dims."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .ops import BinaryOperation, Operation

if TYPE_CHECKING:
    from .tensor import Tensor


class AddOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "add"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.add(args[0], args[1])


class MulOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "mul"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.mul(args[0], args[1])


class SubOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "sub"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.sub(args[0], args[1])


class DivOp(BinaryOperation):
    @property
    def name(self) -> str:
        return "div"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.div(args[0], args[1])


class MatmulOp(Operation):
    """Matmul with 1D promotion handling."""
    
    @property
    def name(self) -> str:
        return "matmul"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        return ops.matmul(args[0], args[1])
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        from .tensor import Tensor
        from . import logical_view_ops as view_ops
        
        x_was_1d = len(x.shape) == 1
        y_was_1d = len(y.shape) == 1
        
        if x_was_1d:
            x = view_ops.unsqueeze(x, axis=0)
        if y_was_1d:
            y = view_ops.unsqueeze(y, axis=-1)
        
        result = super().__call__(x, y)
        
        if x_was_1d and y_was_1d:
            result = view_ops.squeeze(result, axis=-1)
            result = view_ops.squeeze(result, axis=-1)
        elif x_was_1d:
            result = view_ops.squeeze(result, axis=0)
        elif y_was_1d:
            result = view_ops.squeeze(result, axis=-1)
        
        return result


add = AddOp()
mul = MulOp()
sub = SubOp()
div = DivOp()
matmul = MatmulOp()


__all__ = [
    "AddOp", "MulOp", "SubOp", "DivOp", "MatmulOp",
    "add", "mul", "sub", "div", "matmul",
]
