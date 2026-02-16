# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.dtype import DType

from ...core import Tensor
from ...ops.creation import gaussian


def xavier_normal(
    shape: tuple[int, ...],
    *,
    dtype: DType = DType.float32,
    device: str | None = None,
) -> Tensor:
    """Xavier/Glorot normal initializer for dense layers."""
    if len(shape) < 2:
        raise ValueError("xavier_normal expects shape with at least 2 dims")
    fan_in = int(shape[0])
    fan_out = int(shape[1])
    std = (2.0 / float(fan_in + fan_out)) ** 0.5
    return gaussian(shape, mean=0.0, std=std, dtype=dtype, device=device)


def he_normal(
    shape: tuple[int, ...],
    *,
    dtype: DType = DType.float32,
    device: str | None = None,
) -> Tensor:
    """He normal initializer."""
    if len(shape) < 1:
        raise ValueError("he_normal expects shape with at least 1 dim")
    fan_in = int(shape[0])
    std = (2.0 / float(fan_in)) ** 0.5
    return gaussian(shape, mean=0.0, std=std, dtype=dtype, device=device)
