# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from ...core import Tensor
from .. import functional as F
from .base import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)


class Softmax(Module):
    """Apply softmax along a given axis (default: last)."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, axis=self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"
