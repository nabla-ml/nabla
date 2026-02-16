# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...core import Tensor, is_tensor, tree_map
from ...ops.creation import zeros_like
from ...ops.unary import sqrt
from .functional import adamw_step


class Optimizer(ABC):
    """Base class for stateful optimizers backed by pure functional steps."""

    def __init__(self, params: Any) -> None:
        self.params = params

    @abstractmethod
    def step(self, grads: Any) -> Any:
        raise NotImplementedError


class AdamW(Optimizer):
    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0

        self.m = tree_map(lambda p: zeros_like(p) if is_tensor(p) else None, params)
        self.v = tree_map(lambda p: zeros_like(p) if is_tensor(p) else None, params)

    def step(self, grads: Any) -> Any:
        self.step_count += 1

        def _apply(p: Any, g: Any, m: Any, v: Any):
            if is_tensor(p) and is_tensor(g) and is_tensor(m) and is_tensor(v):
                return adamw_step(
                    p,
                    g,
                    m,
                    v,
                    self.step_count,
                    lr=self.lr,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            return p, m, v

        triples = tree_map(_apply, self.params, grads, self.m, self.v)

        def _is_triplet_leaf(x: Any) -> bool:
            return (
                isinstance(x, tuple)
                and len(x) == 3
                and is_tensor(x[0])
                and is_tensor(x[1])
                and is_tensor(x[2])
            )

        self.params = tree_map(lambda t: t[0], triples, is_leaf=_is_triplet_leaf)
        self.m = tree_map(lambda t: t[1], triples, is_leaf=_is_triplet_leaf)
        self.v = tree_map(lambda t: t[2], triples, is_leaf=_is_triplet_leaf)

        return self.params


def adamw_init(params: Any) -> dict[str, Any]:
    """Functional AdamW state init for pytree params."""
    m = tree_map(lambda p: zeros_like(p) if is_tensor(p) else None, params)
    v = tree_map(lambda p: zeros_like(p) if is_tensor(p) else None, params)
    return {"m": m, "v": v, "step": 0}


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
    """Functional AdamW update on pytrees.

    Kept for compatibility and reused by finetuning workloads.
    """
    step = int(state["step"]) + 1

    def _update_m(m: Any, g: Any) -> Any:
        if is_tensor(m) and is_tensor(g):
            return m * beta1 + g * (1.0 - beta1)
        return m

    def _update_v(v: Any, g: Any) -> Any:
        if is_tensor(v) and is_tensor(g):
            return v * beta2 + (g * g) * (1.0 - beta2)
        return v

    m = tree_map(_update_m, state["m"], grads)
    v = tree_map(_update_v, state["v"], grads)

    if bias_correction:
        bias_c1 = 1.0 - (beta1**step)
        bias_c2 = 1.0 - (beta2**step)
    else:
        bias_c1 = 1.0
        bias_c2 = 1.0

    def _apply(p: Any, g: Any, m_t: Any, v_t: Any) -> Any:
        if is_tensor(p) and is_tensor(g) and is_tensor(m_t) and is_tensor(v_t):
            m_hat = m_t / bias_c1
            v_hat = v_t / bias_c2
            update = m_hat / (sqrt(v_hat) + eps)
            if weight_decay != 0.0:
                update = update + p * weight_decay
            return p - update * lr
        return p

    new_params = tree_map(_apply, params, grads, m, v)
    return new_params, {"m": m, "v": v, "step": step}
