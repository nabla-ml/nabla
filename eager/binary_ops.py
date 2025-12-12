# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Binary operations for the eager module."""

from __future__ import annotations

from typing import Any

from max.graph import TensorValue, ops

from .ops import Operation


class AddOp(Operation):
    """Element-wise addition operation."""
    
    @property
    def name(self) -> str:
        return "add"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Add two tensors element-wise."""
        return ops.add(args[0], args[1])


class MulOp(Operation):
    """Element-wise multiplication operation."""
    
    @property
    def name(self) -> str:
        return "mul"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Multiply two tensors element-wise."""
        return ops.mul(args[0], args[1])


class SubOp(Operation):
    """Element-wise subtraction operation."""
    
    @property
    def name(self) -> str:
        return "sub"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Subtract two tensors element-wise."""
        return ops.sub(args[0], args[1])


class DivOp(Operation):
    """Element-wise division operation."""
    
    @property
    def name(self) -> str:
        return "div"
    
    def maxpr(self, *args: TensorValue, **kwargs: Any) -> TensorValue:
        """Divide two tensors element-wise."""
        return ops.div(args[0], args[1])


class MatmulOp(Operation):
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
