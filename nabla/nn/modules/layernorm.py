# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.dtype import DType

from ...core import Tensor
from ...ops.creation import ones, zeros
from .. import functional as F
from .base import Module


class LayerNorm(Module):
    """Apply layer normalization over the last ``len(normalized_shape)`` dimensions.

    Normalises inputs as ``(x - mean) / sqrt(var + eps)`` and then applies
    a learnable per-element affine transform when *elementwise_affine* is
    ``True``.

    Args:
        normalized_shape: Input shape from an expected input of size
            ``(*, normalized_shape[0], ..., normalized_shape[-1])``.
            Can be an ``int`` for the common last-dimension case.
        eps: Value added to the denominator for numerical stability.
            Default: ``1e-5``.
        elementwise_affine: If ``True`` (default), learnable ``weight``
            (initialized to 1) and ``bias`` (initialized to 0) are added.
        dtype: Dtype for weight and bias. Default: ``float32``.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *,
        dtype: DType = DType.float32,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)

        if self.elementwise_affine:
            weight = ones(self.normalized_shape, dtype=dtype)
            weight.requires_grad_(True)
            bias = zeros(self.normalized_shape, dtype=dtype)
            bias.requires_grad_(True)
            self.weight = weight
            self.bias = bias
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        axis = tuple(range(-len(self.normalized_shape), 0))
        return F.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            axis=axis,
        )

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
