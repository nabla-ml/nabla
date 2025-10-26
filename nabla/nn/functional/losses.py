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

import numpy as np
import nabla as nb

__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "huber_loss",
    "cross_entropy_loss",
]


def mean_squared_error(predictions: nb.Tensor, targets: nb.Tensor) -> nb.Tensor:
    diff = predictions - targets
    return nb.mean(diff * diff)


def mean_absolute_error(predictions: nb.Tensor, targets: nb.Tensor) -> nb.Tensor:
    diff = predictions - targets
    return nb.mean(nb.abs(diff))


def huber_loss(predictions: nb.Tensor, targets: nb.Tensor, delta: float = 1.0) -> nb.Tensor:
    diff = nb.abs(predictions - targets)
    quadratic = nb.minimum(diff, delta)
    linear = diff - quadratic
    return nb.mean(0.5 * quadratic**2 + delta * linear)


def cross_entropy_loss(logits: nb.Tensor, targets: nb.Tensor, axis: int = -1) -> nb.Tensor:
    log_probs = nb.log_softmax(logits, axes=axis)
    return -nb.sum(targets * log_probs) / logits.shape[0]
