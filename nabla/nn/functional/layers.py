# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from ...core import Tensor
from ...ops.binary import matmul


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply a linear projection: y = x @ weight + bias."""
    out = matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out
