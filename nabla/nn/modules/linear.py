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

"""Linear (fully-connected) layer implementation."""

from __future__ import annotations

import nabla as nb
from ..module import Module

__all__ = ["Linear"]


class Linear(Module):
    """
    Applies a linear transformation: y = x @ W + b
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds a learnable bias (default: True)
        
    Shape:
        - Input: (*, in_features) where * means any number of dimensions
        - Output: (*, out_features)
        
    Attributes:
        weight: Learnable weights of shape (in_features, out_features)
        bias: Learnable bias of shape (1, out_features) if bias=True
        
    Example:
        >>> layer = Linear(20, 30)
        >>> input = nb.rand((128, 20))
        >>> output = layer(input)
        >>> print(output.shape)
        (128, 30)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weight with Glorot/Xavier uniform initialization
        weight = nb.glorot_uniform((in_features, out_features))
        weight.requires_grad_(True)
        self.weight = weight  # Auto-registered as parameter!
        
        # Initialize bias if requested
        if bias:
            bias_tensor = nb.zeros((1, out_features))
            bias_tensor.requires_grad_(True)
            self.bias = bias_tensor  # Auto-registered as parameter!
    
    def forward(self, x: nb.Tensor) -> nb.Tensor:
        """
        Forward pass of the linear layer.
        
        Args:
            x: Input tensor of shape (*, in_features)
            
        Returns:
            Output tensor of shape (*, out_features)
        """
        out = nb.matmul(x, self.weight)
        if self.use_bias:
            out = out + self.bias
        return out
    
    def __repr__(self) -> str:
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.use_bias})")
