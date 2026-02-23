# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

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
    """Single-tensor SGD update.

    Returns ``(new_param, new_momentum_buffer)``.
    """
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


def sgd_update(
    params: Any,
    grads: Any,
    state: dict[str, Any] | None = None,
    *,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Functional SGD update on pytrees (mirrors ``adamw_update``).

    Parameters
    ----------
    params : pytree
        Current model parameters.
    grads : pytree
        Gradients matching the *params* structure.
    state : dict, optional
        Optimizer state containing ``"momentum_buffers"`` and ``"step"``.
        If *None* a fresh state is created.
    lr, momentum, weight_decay : float
        Standard SGD hyper-parameters.

    Returns
    -------
    (new_params, new_state) : tuple
        Updated parameters and optimizer state, with tensors realized
        according to the global ``Optimizer`` execution policy.
    """
    from ...core import is_tensor, realize_all, tree_leaves, tree_map
    from ...ops.creation import zeros_like

    if state is None:
        if momentum != 0.0:
            bufs = tree_map(lambda p: zeros_like(p) if is_tensor(p) else None, params)
        else:
            bufs = tree_map(lambda p: None, params)
        state = {"momentum_buffers": bufs, "step": 0}

    step = int(state["step"]) + 1

    def _apply(p: Any, g: Any, buf: Any) -> Any:
        if is_tensor(p) and is_tensor(g):
            new_p, new_buf = sgd_step(
                p,
                g,
                buf,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
            return new_p, new_buf
        return p, buf

    pairs = tree_map(_apply, params, grads, state["momentum_buffers"])

    def _is_pair(x: Any) -> bool:
        return isinstance(x, tuple) and len(x) == 2

    new_params = tree_map(lambda t: t[0], pairs, is_leaf=_is_pair)
    new_bufs = tree_map(lambda t: t[1], pairs, is_leaf=_is_pair)
    new_state = {"momentum_buffers": new_bufs, "step": step}

    # Honour the global auto-realization policy
    from .optimizer import Optimizer

    if Optimizer._AUTO_REALIZE_UPDATED_PARAMS:
        to_realize = [t for t in tree_leaves(new_params) if is_tensor(t) and not t.real]
        if to_realize:
            realize_all(*to_realize)

    return new_params, new_state


def adamw_step(
    param: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    step: int | float | Tensor,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    bias_correction: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Single-tensor AdamW update.

    Handles both scalar and tensor ``step`` (the latter is needed inside
    ``@nb.compile`` where the step counter lives as a 0-D tensor).
    """
    import math

    from ...core import is_tensor
    from ...ops.unary import exp

    m_t = m * beta1 + grad * (1.0 - beta1)
    v_t = v * beta2 + (grad * grad) * (1.0 - beta2)

    if bias_correction:
        if is_tensor(step):
            bias_c1 = 1.0 - exp(step * math.log(beta1))
            bias_c2 = 1.0 - exp(step * math.log(beta2))
        else:
            bias_c1 = 1.0 - (beta1 ** step)
            bias_c2 = 1.0 - (beta2 ** step)
    else:
        bias_c1 = 1.0
        bias_c2 = 1.0

    m_hat = m_t / bias_c1
    v_hat = v_t / bias_c2

    update = m_hat / (sqrt(v_hat) + eps)
    if weight_decay != 0.0:
        update = update + param * weight_decay

    new_param = param - update * lr
    return new_param, m_t, v_t
