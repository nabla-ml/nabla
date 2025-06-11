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
"""Learning rate schedules."""

import math


def constant_schedule(initial_lr: float = 0.001) -> callable:
    """Constant learning rate schedule.

    Args:
        initial_lr: The learning rate to maintain

    Returns:
        Function that takes epoch and returns learning rate
    """

    def schedule(epoch: int) -> float:
        return initial_lr

    return schedule


def exponential_decay_schedule(
    initial_lr: float = 0.001,
    decay_factor: float = 0.95,
    decay_every: int = 1000,
) -> callable:
    """Exponential decay learning rate schedule.

    Args:
        initial_lr: Initial learning rate
        decay_factor: Factor to multiply learning rate by
        decay_every: Apply decay every N epochs

    Returns:
        Function that takes epoch and returns learning rate
    """

    def schedule(epoch: int) -> float:
        return initial_lr * (decay_factor ** (epoch // decay_every))

    return schedule


def step_decay_schedule(
    initial_lr: float = 0.001,
    decay_factor: float = 0.1,
    step_size: int = 30,
) -> callable:
    """Step decay learning rate schedule.

    Args:
        initial_lr: Initial learning rate
        decay_factor: Factor to multiply learning rate by at each step
        step_size: Number of epochs between each decay step

    Returns:
        Function that takes epoch and returns learning rate
    """

    def schedule(epoch: int) -> float:
        return initial_lr * (decay_factor ** (epoch // step_size))

    return schedule


def cosine_annealing_schedule(
    initial_lr: float = 0.001,
    min_lr: float = 1e-6,
    period: int = 1000,
) -> callable:
    """Cosine annealing learning rate schedule.

    Args:
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        period: Number of epochs for one complete cosine cycle

    Returns:
        Function that takes epoch and returns learning rate
    """

    def schedule(epoch: int) -> float:
        cycle_position = epoch % period
        cosine_factor = 0.5 * (1 + math.cos(math.pi * cycle_position / period))
        return min_lr + (initial_lr - min_lr) * cosine_factor

    return schedule


def warmup_cosine_schedule(
    initial_lr: float = 0.001,
    warmup_epochs: int = 100,
    total_epochs: int = 1000,
    min_lr: float = 1e-6,
) -> callable:
    """Warmup followed by cosine annealing schedule.

    Args:
        initial_lr: Peak learning rate after warmup
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        min_lr: Minimum learning rate

    Returns:
        Function that takes epoch and returns learning rate
    """

    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * epoch / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (initial_lr - min_lr) * cosine_factor

    return schedule


# Legacy function for backward compatibility
def learning_rate_schedule(
    epoch: int,
    initial_lr: float = 0.001,
    decay_factor: float = 0.95,
    decay_every: int = 1000,
) -> float:
    """Learning rate schedule for complex function learning.

    This is the original function from mlp_train_jit.py for backward compatibility.
    Consider using exponential_decay_schedule instead for new code.

    Args:
        epoch: Current epoch number
        initial_lr: Initial learning rate
        decay_factor: Factor to multiply learning rate by
        decay_every: Apply decay every N epochs

    Returns:
        Learning rate for the current epoch
    """
    return initial_lr * (decay_factor ** (epoch // decay_every))
