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
"""SGD optimizer implementation."""

import nabla as nb


@nb.jit
def sgd_step(
    params: list[nb.Array],
    gradients: list[nb.Array],
    momentum_states: list[nb.Array] | None = None,
    learning_rate: float = 0.01,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False,
) -> tuple[list[nb.Array], list[nb.Array]]:
    """Perform a single SGD optimization step.

    Args:
        params: List of parameter arrays
        gradients: List of gradient arrays (same structure as params)
        momentum_states: List of momentum buffers (None for first step)
        learning_rate: Learning rate
        momentum: Momentum factor
        weight_decay: Weight decay (L2 penalty)
        dampening: Dampening for momentum
        nesterov: Enable Nesterov momentum

    Returns:
        Tuple of (updated_params, updated_momentum_states)
    """
    updated_params = []
    updated_momentum_states = []

    for i, (param, grad) in enumerate(zip(params, gradients, strict=False)):
        # Add weight decay
        if weight_decay != 0:
            grad = grad + weight_decay * param

        # Initialize momentum state if needed
        if momentum_states is None or len(momentum_states) <= i:
            momentum_state = nb.zeros_like(param)
        else:
            momentum_state = momentum_states[i]

        # Update momentum
        if momentum != 0:
            if i == 0 or momentum_states is None:
                # First step or no previous momentum
                buf = grad
            else:
                buf = momentum * momentum_state + (1 - dampening) * grad

            grad = grad + momentum * buf if nesterov else buf

            updated_momentum_states.append(buf)
        else:
            updated_momentum_states.append(momentum_state)

        # Update parameters
        updated_param = param - learning_rate * grad
        updated_params.append(updated_param)

    return updated_params, updated_momentum_states


def init_sgd_state(params: list[nb.Array]) -> list[nb.Array]:
    """Initialize SGD momentum states.

    Args:
        params: List of parameter arrays

    Returns:
        List of zero-initialized momentum states
    """
    return [nb.zeros_like(param) for param in params]


__all__ = ["sgd_step", "init_sgd_state"]
