# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from ...core import Tensor
from ...ops.unary import sqrt


def sgd_step(
    param: Tensor,
    grad: Tensor,
    momentum_buffer: Tensor | None = None,
    *,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
) -> tuple[Tensor, Tensor | None]:
    """Single-tensor SGD update."""
    update = grad
    if weight_decay != 0.0:
        update = update + param * weight_decay

    if momentum != 0.0:
        if momentum_buffer is None:
            momentum_buffer = update
        else:
            momentum_buffer = momentum * momentum_buffer + update
        update = momentum_buffer

    new_param = param - update * lr
    return new_param, momentum_buffer


def adamw_step(
    param: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    step: int,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Single-tensor AdamW update."""
    m_t = m * beta1 + grad * (1.0 - beta1)
    v_t = v * beta2 + (grad * grad) * (1.0 - beta2)

    bias_c1 = 1.0 - (beta1**step)
    bias_c2 = 1.0 - (beta2**step)

    m_hat = m_t / bias_c1
    v_hat = v_t / bias_c2

    update = m_hat / (sqrt(v_hat) + eps)
    if weight_decay != 0.0:
        update = update + param * weight_decay

    new_param = param - update * lr
    return new_param, m_t, v_t
