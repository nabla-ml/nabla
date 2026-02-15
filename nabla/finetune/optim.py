# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

from ..core import Tensor, tree_map
from ..ops.creation import zeros_like
from ..ops.unary import sqrt


def _is_tensor(x: Any) -> bool:
    return isinstance(x, Tensor)


def _iter_leaves(tree: Any):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _iter_leaves(value)
        return
    if isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _iter_leaves(value)
        return
    yield tree


def adamw_init(params: Any) -> dict[str, Any]:
    """Initialize AdamW optimizer state for a pytree of tensors."""
    m = tree_map(lambda p: zeros_like(p) if _is_tensor(p) else None, params)
    v = tree_map(lambda p: zeros_like(p) if _is_tensor(p) else None, params)
    return {
        "m": m,
        "v": v,
        "step": 0,
    }


def adamw_update(
    params: Any,
    grads: Any,
    state: dict[str, Any],
    *,
    lr: float,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Apply one AdamW update on a pytree of tensors."""
    step = int(state["step"]) + 1
    m_prev, v_prev = state["m"], state["v"]

    def _update_m(m: Any, g: Any) -> Any:
        if _is_tensor(m) and _is_tensor(g):
            return m * beta1 + g * (1.0 - beta1)
        return m

    def _update_v(v: Any, g: Any) -> Any:
        if _is_tensor(v) and _is_tensor(g):
            return v * beta2 + (g * g) * (1.0 - beta2)
        return v

    m = tree_map(_update_m, m_prev, grads)
    v = tree_map(_update_v, v_prev, grads)

    if bias_correction:
        bias_c1 = 1.0 - (beta1**step)
        bias_c2 = 1.0 - (beta2**step)
    else:
        bias_c1 = 1.0
        bias_c2 = 1.0

    def _apply(p: Any, g: Any, m_t: Any, v_t: Any) -> Any:
        if _is_tensor(p) and _is_tensor(g) and _is_tensor(m_t) and _is_tensor(v_t):
            m_hat = m_t / bias_c1
            v_hat = v_t / bias_c2
            update = m_hat / (sqrt(v_hat) + eps)
            if weight_decay != 0.0:
                update = update + p * weight_decay
            return p - update * lr
        return p

    new_params = tree_map(_apply, params, grads, m, v)
    return new_params, {
        "m": m,
        "v": v,
        "step": step,
    }
