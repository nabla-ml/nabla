# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .base import UnaryOperation

if TYPE_CHECKING:
    from ..core import Tensor



class ReluOp(UnaryOperation):
    """Rectified Linear Unit (ReLU) activation: max(0, x)."""
    
    @property
    def name(self) -> str:
        return "relu"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply ReLU element-wise."""
        return ops.relu(x)


class SigmoidOp(UnaryOperation):
    """Sigmoid activation function: 1 / (1 + exp(-x))."""
    
    @property
    def name(self) -> str:
        return "sigmoid"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply sigmoid element-wise."""
        return ops.sigmoid(x)
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Sigmoid: ~4 FLOPs per element (neg, exp, add, div)."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return 4.0 * num_elements


class TanhOp(UnaryOperation):
    """Hyperbolic tangent activation."""
    
    @property
    def name(self) -> str:
        return "tanh"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply tanh element-wise."""
        return ops.tanh(x)
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Tanh: ~6 FLOPs per element (2 exp, 2 add/sub, 1 div)."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return 6.0 * num_elements


class ExpOp(UnaryOperation):
    """Exponential function: e^x."""
    
    @property
    def name(self) -> str:
        return "exp"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply exp element-wise."""
        return ops.exp(x)


class NegOp(UnaryOperation):
    """Negation: -x."""
    
    @property
    def name(self) -> str:
        return "neg"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply negation element-wise."""
        return ops.negate(x)


class AbsOp(UnaryOperation):
    """Absolute value: |x|."""
    
    @property
    def name(self) -> str:
        return "abs"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply abs element-wise."""
        return ops.abs(x)


class SoftmaxOp(UnaryOperation):
    """Softmax activation function: exp(x) / sum(exp(x))."""
    
    @property
    def name(self) -> str:
        return "softmax"
    
    def __call__(self, x: "Tensor", axis: int = -1) -> "Tensor":
        """Apply softmax along specified axis."""
        return super().__call__(x, axis=axis)
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply softmax using MAX's native softmax."""
        axis = kwargs.get("axis", -1)
        return ops.softmax(x, axis=axis)
    
    def compute_cost(self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> float:
        """Softmax: ~3 FLOPs per element (exp, sum, div)."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return 3.0 * num_elements


relu = ReluOp()
sigmoid = SigmoidOp()
tanh = TanhOp()
exp = ExpOp()
neg = NegOp()
abs = AbsOp()
softmax = SoftmaxOp()


__all__ = [
    "ReluOp",
    "SigmoidOp",
    "TanhOp",
    "ExpOp",
    "NegOp",
    "AbsOp",
    "SoftmaxOp",
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "neg",
    "abs",
    "softmax",
]
