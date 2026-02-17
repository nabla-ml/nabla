# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Pure functional loss functions."""

from __future__ import annotations

from ...core import Tensor
from ...ops.reduction import mean, reduce_sum
from ...ops.unary import logsoftmax


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error loss."""
    diff = predictions - targets
    return mean(diff * diff)


def cross_entropy_loss(logits: Tensor, targets: Tensor, axis: int = -1) -> Tensor:
    """Cross-entropy loss.

    Parameters
    ----------
    logits : Tensor
        Unnormalized predictions of shape ``(batch, ..., num_classes)``.
    targets : Tensor
        Either one-hot labels with the same shape as *logits*, or
        integer class indices of shape ``(batch, ...)``.  When the rank of
        *targets* is one less than *logits* the targets are treated as
        class indices and converted to one-hot internally.
    axis : int
        The class axis for softmax (default ``-1``).
    """
    log_probs = logsoftmax(logits, axis=axis)

    # Detect integer-target mode: targets has one fewer dimension than logits
    if len(targets.shape) < len(logits.shape):
        targets = _one_hot(
            targets, num_classes=int(logits.shape[axis]), dtype=logits.dtype
        )

    batch = int(logits.shape[0])
    return -reduce_sum(targets * log_probs) / float(batch)


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _one_hot(indices: Tensor, num_classes: int, dtype=None) -> Tensor:
    """Convert integer indices to one-hot vectors.

    Parameters
    ----------
    indices : Tensor  shape ``(*)``
        Integer class indices.
    num_classes : int
        Number of classes.

    Returns
    -------
    Tensor of shape ``(*, num_classes)`` with 1s at the index positions.
    """
    from ...ops.comparison import equal
    from ...ops.creation import arange
    from ...ops.unary import cast
    from ...ops.view import unsqueeze

    # indices: (*), classes: (num_classes,)
    classes = arange(0, num_classes, dtype=indices.dtype)
    # Broadcast compare: (*, 1) == (num_classes,) -> (*, num_classes)
    mask = equal(unsqueeze(indices, -1), classes)
    if dtype is not None:
        mask = cast(mask, dtype)
    return mask
