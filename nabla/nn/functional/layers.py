# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from ...core import Tensor
from ...ops.binary import matmul
from ...ops.reduction import mean
from ...ops.unary import rsqrt


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply a linear projection: y = x @ weight + bias."""
    out = matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out


def layer_norm(
    x: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
    axis: int | tuple[int, ...] = -1,
) -> Tensor:
    """Apply layer normalization over one or more axes."""
    mu = mean(x, axis=axis, keepdims=True)
    centered = x - mu
    var = mean(centered * centered, axis=axis, keepdims=True)
    normalized = centered * rsqrt(var + eps)

    out = normalized
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out
