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
"""AdamW optimizer implementation."""

import numpy as np
import nabla as nb


@nb.jit
def adamw_step(
    params: list[nb.Array],
    gradients: list[nb.Array],
    m_states: list[nb.Array],
    v_states: list[nb.Array],
    step: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[nb.Array], list[nb.Array], list[nb.Array]]:
    """JIT-compiled AdamW optimizer step with weight decay.
    
    AdamW decouples weight decay from the gradient-based update, applying
    weight decay directly to parameters rather than adding L2 regularization
    to the loss function.
    
    Args:
        params: List of parameter arrays
        gradients: List of gradient arrays (same structure as params)
        m_states: List of first moment estimates
        v_states: List of second moment estimates  
        step: Current step number (for bias correction)
        learning_rate: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient
        
    Returns:
        Tuple of (updated_params, updated_m_states, updated_v_states)
    """
    updated_params = []
    updated_m = []
    updated_v = []

    for param, grad, m, v in zip(params, gradients, m_states, v_states, strict=False):
        # Update biased first and second moment estimates
        new_m = beta1 * m + (1.0 - beta1) * grad
        new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

        # Bias correction
        m_hat = new_m / (1.0 - beta1**step)
        v_hat = new_v / (1.0 - beta2**step)
        
        # AdamW update: weight decay applied directly to parameters
        new_param = param * (1.0 - weight_decay * learning_rate) - learning_rate * m_hat / (nb.sqrt(v_hat) + eps)

        updated_params.append(new_param)
        updated_m.append(new_m)
        updated_v.append(new_v)

    return updated_params, updated_m, updated_v


def init_adamw_state(params: list[nb.Array]) -> tuple[list[nb.Array], list[nb.Array]]:
    """Initialize AdamW optimizer state.
    
    Args:
        params: List of parameter arrays
        
    Returns:
        Tuple of (m_states, v_states) - first and second moment estimates
    """
    m_states = []
    v_states = []
    for param in params:
        # Initialize first and second moments to zero
        m_np = np.zeros_like(param.to_numpy())
        v_np = np.zeros_like(param.to_numpy())
        m_states.append(nb.Array.from_numpy(m_np))
        v_states.append(nb.Array.from_numpy(v_np))
    return m_states, v_states