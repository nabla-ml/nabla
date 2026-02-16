# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from ...core import Tensor
from ...ops.unary import gelu, relu
from .base import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)
