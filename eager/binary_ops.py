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

All binary ops implement jvp_rule for forward-mode autodiff:
- add(x, y): d(x+y) = dx + dy
- sub(x, y): d(x-y) = dx - dy  
- mul(x, y): d(x*y) = y*dx + x*dy
- div(x, y): d(x/y) = dx/y - x*dy/y²
- matmul(x, y): d(x@y) = dx@y + x@dy
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from max.graph import TensorValue, ops

from .ops import Operation

if TYPE_CHECKING:
    from .tensor import Tensor


class AddOp(Operation):
    """Element-wise addition operation."""
    
    @property
    def name(self) -> str:
        return "add"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Add two tensors element-wise."""
        return ops.add(args[0], args[1])
    
    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """d(x + y) = dx + dy"""
        dx, dy = tangents[0], tangents[1]
        if dx is None and dy is None:
            return None
        if dx is None:
            return dy
        if dy is None:
            return dx
        return dx + dy


class MulOp(Operation):
    """Element-wise multiplication operation."""
    
    @property
    def name(self) -> str:
        return "mul"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Multiply two tensors element-wise."""
        return ops.mul(args[0], args[1])
    
    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """d(x * y) = y*dx + x*dy"""
        x, y = primals[0], primals[1]
        dx, dy = tangents[0], tangents[1]
        
        result = None
        if dx is not None:
            result = y * dx
        if dy is not None:
            term = x * dy
            result = term if result is None else result + term
        return result


class SubOp(Operation):
    """Element-wise subtraction operation."""
    
    @property
    def name(self) -> str:
        return "sub"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Subtract two tensors element-wise."""
        return ops.sub(args[0], args[1])
    
    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """d(x - y) = dx - dy"""
        dx, dy = tangents[0], tangents[1]
        if dx is None and dy is None:
            return None
        if dx is None:
            # 0 - dy = -dy
            from .tensor import Tensor
            zero = Tensor.zeros(dy.shape, dtype=dy.dtype, device=dy.device)
            return zero - dy
        if dy is None:
            return dx
        return dx - dy


class DivOp(Operation):
    """Element-wise division operation."""
    
    @property
    def name(self) -> str:
        return "div"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Divide two tensors element-wise."""
        return ops.div(args[0], args[1])
    
    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """d(x / y) = dx/y - x*dy/y²"""
        x, y = primals[0], primals[1]
        dx, dy = tangents[0], tangents[1]
        
        result = None
        if dx is not None:
            result = dx / y
        if dy is not None:
            # -x * dy / y² = -output * dy / y
            term = output * dy / y
            if result is None:
                from .tensor import Tensor
                zero = Tensor.zeros(term.shape, dtype=term.dtype, device=term.device)
                result = zero - term
            else:
                result = result - term
        return result


class MatmulOp(Operation):
    """Matrix multiplication operation."""
    
    @property
    def name(self) -> str:
        return "matmul"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Matrix multiplication of two tensors."""
        return ops.matmul(args[0], args[1])
    
    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        """d(x @ y) = dx @ y + x @ dy"""
        x, y = primals[0], primals[1]
        dx, dy = tangents[0], tangents[1]
        
        result = None
        if dx is not None:
            result = dx @ y
        if dy is not None:
            term = x @ dy
            result = term if result is None else result + term
        return result


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
