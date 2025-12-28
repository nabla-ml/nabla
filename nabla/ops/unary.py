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

"""Unary operations for the nabla module.

All unary ops inherit from UnaryOperation ABC which handles:
- Preserving batch_dims on output for vmap support
- JVP mode propagation

Autodiff rules (vjp_rule, jvp_rule) are not yet implemented - they will
be added when the autodiff infrastructure is tested and ready.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .operation import UnaryOperation

if TYPE_CHECKING:
    from ..core.tensor import Tensor


class ReluOp(UnaryOperation):
    """Rectified Linear Unit (ReLU) activation.
    
    relu(x) = max(0, x)
    """
    
    @property
    def name(self) -> str:
        return "relu"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply ReLU element-wise."""
        return ops.relu(x)


class SigmoidOp(UnaryOperation):
    """Sigmoid activation function.
    
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    
    @property
    def name(self) -> str:
        return "sigmoid"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply sigmoid element-wise."""
        return ops.sigmoid(x)


class TanhOp(UnaryOperation):
    """Hyperbolic tangent activation.
    
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    
    @property
    def name(self) -> str:
        return "tanh"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply tanh element-wise."""
        return ops.tanh(x)


class ExpOp(UnaryOperation):
    """Exponential function.
    
    exp(x) = e^x
    """
    
    @property
    def name(self) -> str:
        return "exp"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply exp element-wise."""
        return ops.exp(x)


class NegOp(UnaryOperation):
    """Negation.
    
    neg(x) = -x
    """
    
    @property
    def name(self) -> str:
        return "neg"
    
    def maxpr(self, x: TensorValue, **kwargs: Any) -> TensorValue:
        """Apply negation element-wise."""
        return ops.negate(x)


# ===== Singleton instances exposed as functions =====

relu = ReluOp()
sigmoid = SigmoidOp()
tanh = TanhOp()
exp = ExpOp()
neg = NegOp()


__all__ = [
    "ReluOp",
    "SigmoidOp",
    "TanhOp",
    "ExpOp",
    "NegOp",
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "neg",
]
