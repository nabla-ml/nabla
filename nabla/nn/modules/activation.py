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
