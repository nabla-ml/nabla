# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

from max.dtype import DType

from ...core import Tensor, tree_map
from ...ops.creation import gaussian, zeros


def init_lora_adapter(
    weight: Tensor,
    rank: int,
    init_std: float = 0.01,
    dtype: DType | None = None,
) -> dict[str, Tensor]:
    """Initialise LoRA adapter matrices ``A`` and ``B`` for a 2D weight.

    Following Hu et al. (2021), ``A`` is initialised with Gaussian noise and
    ``B`` is zero-initialised so the adapter adds zero at the start of training.

    Args:
        weight: The frozen 2D weight tensor to adapt. Shape ``(in, out)``.
        rank: Intrinsic rank of the low-rank decomposition. Must be > 0.
        init_std: Standard deviation for initialising ``A``. Default: ``0.01``.
        dtype: Optional dtype override. Defaults to *weight*'s dtype.

    Returns:
        ``{'A': Tensor(in, rank), 'B': Tensor(rank, out)}``
    """
    if len(weight.shape) != 2:
        raise ValueError("init_lora_adapter expects a 2D weight tensor")
    if rank <= 0:
        raise ValueError("rank must be > 0")

    in_features, out_features = int(weight.shape[0]), int(weight.shape[1])
    adapter_dtype = dtype or weight.dtype

    A = gaussian(
        (in_features, rank),
        mean=0.0,
        std=init_std,
        dtype=adapter_dtype,
        device=weight.device,
    )
    B = zeros((rank, out_features), dtype=adapter_dtype, device=weight.device)
    return {"A": A, "B": B}


def lora_delta(adapter: dict[str, Tensor], alpha: float = 1.0) -> Tensor:
    """Compute the scaled LoRA weight update: ``(alpha / rank) * A @ B``.

    Args:
        adapter: Dict with keys ``'A'`` ``(in, rank)`` and ``'B'`` ``(rank, out)``.
        alpha: Scaling factor. Default: ``1.0``.

    Returns:
        Delta tensor of shape ``(in, out)``.
    """
    A = adapter["A"]
    B = adapter["B"]
    rank = int(A.shape[1])
    scale = alpha / float(rank)
    return (A @ B) * scale


def lora_linear(
    x: Tensor,
    frozen_weight: Tensor,
    adapter: dict[str, Tensor],
    alpha: float = 1.0,
) -> Tensor:
    """Linear projection with frozen path + LoRA adapter path."""
    return (x @ frozen_weight) + (x @ lora_delta(adapter, alpha=alpha))


def merge_lora_weight(
    frozen_weight: Tensor, adapter: dict[str, Tensor], alpha: float = 1.0
) -> Tensor:
    """Merge the LoRA adapter into the frozen weight: ``W_merged = W + delta``.

    Args:
        frozen_weight: Original frozen weight tensor.
        adapter: LoRA adapter dict (see :func:`init_lora_adapter`).
        alpha: Scaling factor for the adapter. Default: ``1.0``.

    Returns:
        Merged weight tensor with the same shape as *frozen_weight*.
    """
    return frozen_weight + lora_delta(adapter, alpha=alpha)


def unmerge_lora_weight(
    merged_weight: Tensor, adapter: dict[str, Tensor], alpha: float = 1.0
) -> Tensor:
    """Recover the original frozen weight by subtracting the LoRA delta.

    Args:
        merged_weight: Previously merged weight tensor.
        adapter: LoRA adapter dict used during merging.
        alpha: Scaling factor used during merging. Default: ``1.0``.

    Returns:
        Recovered frozen weight tensor.
    """
    return merged_weight - lora_delta(adapter, alpha=alpha)


def tree_lora_delta(
    adapters: Any,
    alpha: float = 1.0,
    *,
    is_leaf: Any = None,
) -> Any:
    """Map a pytree of LoRA adapter dicts to their low-rank deltas."""

    def _to_delta(leaf: Any) -> Any:
        if isinstance(leaf, dict) and "A" in leaf and "B" in leaf:
            return lora_delta(leaf, alpha=alpha)
        return leaf

    return tree_map(_to_delta, adapters, is_leaf=is_leaf)
