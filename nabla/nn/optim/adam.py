# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Adam optimizer implementation."""

import nabla as nb


@nb.jit
def adam_step(
    params: list[nb.Tensor],
    gradients: list[nb.Tensor],
    m_states: list[nb.Tensor],
    v_states: list[nb.Tensor],
    step: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    maximize: bool = False,
) -> tuple[list[nb.Tensor], list[nb.Tensor], list[nb.Tensor]]:
    """Perform a single Adam optimization step.

    Args:
        params: List of parameter tensors
        gradients: List of gradient tensors (same structure as params)
        m_states: List of first moment estimates
        v_states: List of second moment estimates
        step: Current step number (for bias correction)
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
        eps: Small constant for numerical stability
        weight_decay: Weight decay (L2 penalty)
        amsgrad: Whether to use AMSGrad variant
        maximize: Maximize instead of minimize

    Returns:
        Tuple of (updated_params, updated_m_states, updated_v_states)
    """
    updated_params = []
    updated_m_states = []
    updated_v_states = []

    # Bias correction terms
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    for param, grad, m_state, v_state in zip(
        params, gradients, m_states, v_states, strict=False
    ):
        # Add weight decay
        if weight_decay != 0:
            grad = grad + weight_decay * param

        # Maximize by negating gradients
        if maximize:
            grad = -grad

        # Update biased first moment estimate
        m_new = beta1 * m_state + (1 - beta1) * grad

        # Update biased second raw moment estimate
        v_new = beta2 * v_state + (1 - beta2) * (grad * grad)

        # Compute bias-corrected first moment estimate
        m_hat = m_new / bias_correction1

        # Compute bias-corrected second raw moment estimate
        v_hat = v_new / bias_correction2

        # Update parameters
        denom = nb.sqrt(v_hat) + eps
        step_size = learning_rate

        updated_param = param - step_size * m_hat / denom

        updated_params.append(updated_param)
        updated_m_states.append(m_new)
        updated_v_states.append(v_new)

    return updated_params, updated_m_states, updated_v_states


def init_adam_state(params: list[nb.Tensor]) -> tuple[list[nb.Tensor], list[nb.Tensor]]:
    """Initialize Adam optimizer states.

    Args:
        params: List of parameter tensors

    Returns:
        Tuple of (m_states, v_states) - zero-initialized moment estimates
    """
    m_states = [nb.zeros_like(param) for param in params]
    v_states = [nb.zeros_like(param) for param in params]
    return m_states, v_states


__all__ = ["adam_step", "init_adam_state"]
