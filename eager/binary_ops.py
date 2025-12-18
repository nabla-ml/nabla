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

"""Binary operations for the eager module.

All binary ops inherit from BinaryOperation ABC which handles:
- batch_dims-aware broadcasting for vmap support
- Explicit unsqueeze+broadcast for traced tensors (correct gradient shapes)

Autodiff rules (vjp_rule, jvp_rule) are not yet implemented - they will
be added when the autodiff infrastructure is tested and ready.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .ops import BinaryOperation

if TYPE_CHECKING:
    from .tensor import Tensor


class AddOp(BinaryOperation):
    """Element-wise addition operation."""
    
    @property
    def name(self) -> str:
        return "add"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Add two tensors element-wise."""
        return ops.add(args[0], args[1])


class MulOp(BinaryOperation):
    """Element-wise multiplication operation."""
    
    @property
    def name(self) -> str:
        return "mul"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Multiply two tensors element-wise."""
        return ops.mul(args[0], args[1])


class SubOp(BinaryOperation):
    """Element-wise subtraction operation."""
    
    @property
    def name(self) -> str:
        return "sub"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Subtract two tensors element-wise."""
        return ops.sub(args[0], args[1])


class DivOp(BinaryOperation):
    """Element-wise division operation."""
    
    @property
    def name(self) -> str:
        return "div"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Divide two tensors element-wise."""
        return ops.div(args[0], args[1])


class MatmulOp(BinaryOperation):
    """Matrix multiplication operation."""
    
    @property
    def name(self) -> str:
        return "matmul"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Matrix multiplication of two tensors."""
        return ops.matmul(args[0], args[1])


# ===== Singleton instances exposed as functions =====

add = AddOp()
mul = MulOp()
sub = SubOp()
div = DivOp()
matmul = MatmulOp()


__all__ = [
    "AddOp",
    "MulOp", 
    "SubOp",
    "DivOp",
    "MatmulOp",
    "add",
    "mul",
    "sub",
    "div",
    "matmul",
]
