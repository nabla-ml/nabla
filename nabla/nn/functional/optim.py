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

import nabla as nb

__all__ = ["adam_step", "sgd_step"]


def sgd_step(
    param: nb.Tensor,
    grad: nb.Tensor,
    momentum_buffer: nb.Tensor | None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
) -> tuple[nb.Tensor, nb.Tensor]:
    """Functional SGD step.

    Returns:
        A tuple of (new_param, new_momentum_buffer)
    """
    if weight_decay != 0:
        grad = grad + weight_decay * param

    if momentum != 0:
        if momentum_buffer is None:
            buf = grad
        else:
            buf = momentum * momentum_buffer + grad
        grad = buf
        new_momentum_buffer = buf
    else:
        new_momentum_buffer = momentum_buffer

    new_param = param - lr * grad
    return new_param, new_momentum_buffer



def adam_step(
    param: nb.Tensor,
    grad: nb.Tensor,
    exp_avg: nb.Tensor,
    exp_avg_sq: nb.Tensor,
    step: int,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> tuple[nb.Tensor, nb.Tensor, nb.Tensor]:
    """Functional Adam step.

    Returns:
        A tuple of (new_param, new_exp_avg, new_exp_avg_sq)
    """
    if weight_decay != 0:
        grad = grad + weight_decay * param

    new_exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    new_exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    step_size = lr / bias_correction1
    
    denom = nb.sqrt(new_exp_avg_sq) / (bias_correction2 ** 0.5) + eps

    new_param = param - step_size * new_exp_avg / denom

    return new_param, new_exp_avg, new_exp_avg_sq
