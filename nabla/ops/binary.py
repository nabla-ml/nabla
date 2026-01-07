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
        """Matmul: supports both batched and broadcast cases.
        
        Standard: (batch, m, k) @ (batch, k, n) -> (batch, m, n)
        Broadcast: (batch, m, k) @ (k, n) -> (batch, m, n)  # weights no batch
        """
        from ..sharding.propagation import matmul_template, broadcast_matmul_template
        
        a_rank = len(input_shapes[0])
        b_rank = len(input_shapes[1])
        a_rank = len(input_shapes[0])
        b_rank = len(input_shapes[1])
        
        # Infer output rank from inputs if not provided
        if output_shapes:
            out_rank = len(output_shapes[0])
        else:
            # Matmul logic: output rank = max(a_rank, b_rank) approx (modulo broadcasting)
            # Actually, standard matmul (batch..., m, k) @ (batch..., k, n) -> (batch..., m, n)
            # Input ranks: B+2, B+2 -> B+2 (same).
            # Broadcast matmul: (batch..., m, k) @ (k, n) -> (batch..., m, n). Rank: B+2, 2 -> B+2.
            # So out_rank matches whichever input has batch dims (larger rank).
            out_rank = max(a_rank, b_rank)
        
        # If ranks differ, use broadcast template (or standard if ranks match)
        if a_rank != b_rank:
            # broadcast_matmul_template signature needs out_rank for factor generation
            return broadcast_matmul_template(a_rank, b_rank, out_rank).instantiate(
                input_shapes, output_shapes
            )
        
        # Same rank: use standard template
        batch_dims = a_rank - 2
        return matmul_template(batch_dims).instantiate(input_shapes, output_shapes)
    
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
