# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

from max.dtype import DType

from ..core import Tensor, tree_map
from ..ops.creation import gaussian, zeros


def init_lora_adapter(
    weight: Tensor,
    rank: int,
    init_std: float = 0.01,
    dtype: DType | None = None,
) -> dict[str, Tensor]:
    """Initialize LoRA adapter matrices for a 2D linear weight.

    Args:
        weight: Frozen linear weight with shape (in_features, out_features).
        rank: LoRA rank.
        init_std: Stddev for A initialization.
        dtype: Optional dtype override for adapter tensors.

    Returns:
        Dict with tensors: {"A": (in_features, rank), "B": (rank, out_features)}
    """
    if len(weight.shape) != 2:
        raise ValueError("init_lora_adapter expects a 2D weight tensor")
    if rank <= 0:
        raise ValueError("rank must be > 0")

    in_features, out_features = int(weight.shape[0]), int(weight.shape[1])
    adapter_dtype = dtype or weight.dtype

    A = gaussian(
        (in_features, rank), mean=0.0, std=init_std, dtype=adapter_dtype, device=weight.device
    )
    B = zeros((rank, out_features), dtype=adapter_dtype, device=weight.device)
    return {"A": A, "B": B}


def lora_delta(adapter: dict[str, Tensor], alpha: float = 1.0) -> Tensor:
    """Compute scaled LoRA low-rank delta: (alpha / rank) * (A @ B)."""
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
    """Return merged weight: W + (alpha/r) * A @ B."""
    return frozen_weight + lora_delta(adapter, alpha=alpha)


def unmerge_lora_weight(
    merged_weight: Tensor, adapter: dict[str, Tensor], alpha: float = 1.0
) -> Tensor:
    """Recover frozen weight from merged weight and adapter."""
    return merged_weight - lora_delta(adapter, alpha=alpha)


def tree_lora_delta(
    adapters: Any,
    alpha: float = 1.0,
    *,
    is_leaf: Any = None,
) -> Any:
    """Map a pytree of LoRA adapter dicts to their low-rank deltas.

    Non-adapter leaves are passed through unchanged.
    """

    def _to_delta(leaf: Any) -> Any:
        if isinstance(leaf, dict) and "A" in leaf and "B" in leaf:
            return lora_delta(leaf, alpha=alpha)
        return leaf

    return tree_map(_to_delta, adapters, is_leaf=is_leaf)
