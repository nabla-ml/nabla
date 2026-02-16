# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from ...core import Tensor
from ...ops.reduction import mean, reduce_sum
from ...ops.unary import logsoftmax


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error loss."""
    diff = predictions - targets
    return mean(diff * diff)


def cross_entropy_loss(logits: Tensor, targets: Tensor, axis: int = -1) -> Tensor:
    """Cross-entropy from logits and target probabilities/one-hot labels."""
    log_probs = logsoftmax(logits, axis=axis)
    batch = int(logits.shape[0])
    return -reduce_sum(targets * log_probs) / float(batch)
