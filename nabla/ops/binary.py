# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Updated Binary Operations
# ===----------------------------------------------------------------------=== #

"""Binary ops using updated Operation base with auto batch_dims."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .operation import BinaryOperation, Operation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


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
        from ..core.tensor import Tensor
        from . import view as view_ops
        from .operation import ensure_tensor
        
        # Ensure both inputs are Tensors (converts scalars/arrays)
        x = ensure_tensor(x)
        y = ensure_tensor(y)
        
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
    
    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
    ) -> Any:
        """Matmul: (batch..., m, k) @ (batch..., k, n) -> (batch..., m, n).
        
        k is the contracting factor (appears only in inputs).
        Also handles broadcast case where one input lacks batch dims.
        """
        from ..sharding.propagation import OpShardingRuleTemplate
        return OpShardingRuleTemplate.parse("... m k, ... k n -> ... m n", input_shapes).instantiate(
            input_shapes, output_shapes
        )
    
    # NOTE: No custom _infer_output_sharding needed - the generic factor-based
    # propagation handles matmul correctly via sharding_rule() above.



add = AddOp()
mul = MulOp()
sub = SubOp()
div = DivOp()
matmul = MatmulOp()


__all__ = [
    "AddOp", "MulOp", "SubOp", "DivOp", "MatmulOp",
    "add", "mul", "sub", "div", "matmul",
]
