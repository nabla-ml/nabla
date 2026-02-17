# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

import numpy as np
from max.dtype import DType

from ...core import Tensor
from ...ops.unary import cast
from ...ops.view import gather, reshape, unsqueeze
from .lora import lora_delta

NF4_CODEBOOK = np.array(
    [
        -1.0000,
        -0.6962,
        -0.5251,
        -0.3949,
        -0.2844,
        -0.1848,
        -0.0911,
        0.0000,
        0.0796,
        0.1609,
        0.2461,
        0.3379,
        0.4407,
        0.5626,
        0.7230,
        1.0000,
    ],
    dtype=np.float32,
)


def quantize_nf4(weight: Tensor, block_size: int = 64) -> dict[str, Any]:
    """Quantize a 2D weight to NF4 indices + per-block scales."""
    if len(weight.shape) != 2:
        raise ValueError("quantize_nf4 expects a 2D weight tensor")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    w_np = weight.to_numpy().astype(np.float32)
    original_shape = tuple(int(d) for d in w_np.shape)
    flat = w_np.reshape(-1)
    original_numel = int(flat.size)

    pad_len = (block_size - (original_numel % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))

    padded_numel = int(flat.size)
    blocks = flat.reshape(-1, block_size)

    scales = np.max(np.abs(blocks), axis=-1, keepdims=True)
    scales[scales == 0] = 1.0
    normalized = blocks / scales

    diffs = np.abs(normalized.reshape(-1, 1) - NF4_CODEBOOK.reshape(1, -1))
    indices = np.argmin(diffs, axis=-1).astype(np.uint8)

    return {
        "indices": Tensor.from_dlpack(indices),
        "scales": Tensor.from_dlpack(scales.reshape(-1).astype(np.float32)),
        "original_shape": original_shape,
        "original_numel": original_numel,
        "padded_numel": padded_numel,
        "block_size": int(block_size),
    }


def dequantize_nf4(
    qweight: dict[str, Any],
    *,
    dtype: DType = DType.float32,
) -> Tensor:
    """Dequantize NF4 weight dict back to dense tensor using Nabla ops."""
    indices = qweight["indices"]
    scales = qweight["scales"]
    original_shape = qweight["original_shape"]
    original_numel = int(qweight["original_numel"])
    padded_numel = int(qweight["padded_numel"])
    block_size = int(qweight["block_size"])

    num_blocks = padded_numel // block_size

    codebook = Tensor.from_dlpack(NF4_CODEBOOK).to(dtype)
    gathered = gather(codebook, cast(indices, DType.int32), axis=0)
    blocks = reshape(gathered, (num_blocks, block_size))
    scaled = blocks * unsqueeze(scales.to(dtype), axis=1)
    flat = reshape(scaled, (padded_numel,))
    trimmed = flat[:original_numel]
    return reshape(trimmed, original_shape)


def qlora_linear(
    x: Tensor,
    qweight: dict[str, Any],
    adapter: dict[str, Tensor],
    *,
    alpha: float = 1.0,
    compute_dtype: DType = DType.float32,
) -> Tensor:
    """QLoRA-style linear layer using frozen NF4 weight + LoRA adapter."""
    w = dequantize_nf4(qweight, dtype=compute_dtype)
    x_compute = x.to(compute_dtype)
    delta = lora_delta(adapter, alpha=alpha).to(compute_dtype)
    return (x_compute @ w) + (x_compute @ delta)
