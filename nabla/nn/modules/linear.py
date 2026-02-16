# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.dtype import DType

from ...core import Tensor
from ...ops.creation import zeros
from .. import functional as F
from .base import Module


class Linear(Module):
    """Applies y = x @ W + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        dtype: DType = DType.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = F.xavier_normal((in_features, out_features), dtype=dtype)
        weight.requires_grad_(True)
        self.weight = weight

        if bias:
            b = zeros((1, out_features), dtype=dtype)
            b.requires_grad_(True)
            self.bias = b
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
