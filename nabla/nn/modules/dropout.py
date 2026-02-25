# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from ...core import Tensor
from .. import functional as F
from .base import Module


class Dropout(Module):
    """Randomly zero elements of the input with probability *p* (Bernoulli dropout).

    Elements that are not zeroed are scaled by ``1 / (1 - p)`` (inverted dropout)
    so that the expected value of each element is unchanged. Set to ``eval()``
    mode to disable dropout during inference.

    Args:
        p: Probability of an element being zeroed. Default: ``0.5``.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, p=self.p, training=self._training)

    def extra_repr(self) -> str:
        return f"p={self.p}"
