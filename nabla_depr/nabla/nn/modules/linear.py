# ===----------------------------------------------------------------------=== #
# Nabla 2025
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

from __future__ import annotations

import nabla as nb
from .base import Module
from ..functional import layers as F

__all__ = ["Linear"]


class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight with Glorot/Xavier uniform initialization
        weight = nb.glorot_uniform((in_features, out_features))
        weight.requires_grad_(True)
        self.weight = weight
        
        if bias:
            bias_tensor = nb.zeros((1, out_features))
            bias_tensor.requires_grad_(True)
            self.bias = bias_tensor
        else:
            self.bias = None

    def forward(self, x: nb.Tensor) -> nb.Tensor:
        return F.linear_forward(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
