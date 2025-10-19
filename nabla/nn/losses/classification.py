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

"""Classification loss functions."""

import numpy as np

import nabla as nb


def cross_entropy_loss(logits: nb.Tensor, targets: nb.Tensor, axis: int = -1) -> nb.Tensor:
    """Compute cross-entropy loss between logits and targets.

    Args:
        logits: Raw model outputs (before softmax) [batch_size, num_classes]
        targets: One-hot encoded targets [batch_size, num_classes]
        axis: Axis along which to compute softmax

    Returns:
        Scalar loss value
    """
    from ...ops.binary import mul
    from ...ops.reduce import sum as tensor_sum
    from ...ops.special import logsumexp

    # Compute log probabilities using logsumexp for numerical stability
    # log_softmax(x) = x - logsumexp(x)
    log_sum_exp = logsumexp(logits, axis=axis, keep_dims=True)
    log_probs = logits - log_sum_exp

    # Cross-entropy: -sum(targets * log_probs)
    cross_entropy = -tensor_sum(mul(targets, log_probs), axes=axis)

    # Average over batch
    batch_size = nb.tensor([np.float32(logits.shape[0])])
    return tensor_sum(cross_entropy) / batch_size


def sparse_cross_entropy_loss(
    logits: nb.Tensor, targets: nb.Tensor, axis: int = -1
) -> nb.Tensor:
    """Compute cross-entropy loss with integer targets.

    Args:
        logits: Raw model outputs [batch_size, num_classes]
        targets: Integer class indices [batch_size]
        axis: Axis along which to compute softmax

    Returns:
        Scalar loss value
    """
    # Convert targets to one-hot encoding
    num_classes = logits.shape[axis]
    batch_size = targets.shape[0]

    # Create one-hot encoding
    targets_np = targets.to_numpy().astype(np.int32)
    one_hot_np = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot_np[np.arange(batch_size), targets_np] = 1.0

    one_hot_targets = nb.Tensor.from_numpy(one_hot_np)

    return cross_entropy_loss(logits, one_hot_targets, axis=axis)


def binary_cross_entropy_loss(
    predictions: nb.Tensor, targets: nb.Tensor, eps: float = 1e-7
) -> nb.Tensor:
    """Compute binary cross-entropy loss.

    Args:
        predictions: Model predictions (after sigmoid) [batch_size]
        targets: Binary targets (0 or 1) [batch_size]
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value
    """
    from ...ops.binary import mul, sub
    from ...ops.creation import full_like
    from ...ops.reduce import mean
    from ...ops.unary import log

    # Clamp predictions to avoid log(0)
    eps_tensor = full_like(predictions, eps)
    one_minus_eps = full_like(predictions, 1.0 - eps)

    # predictions = clamp(predictions, eps, 1-eps)
    predictions_clamped = nb.maximum(predictions, eps_tensor)
    predictions_clamped = nb.minimum(predictions_clamped, one_minus_eps)

    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    log_p = log(predictions_clamped)
    log_one_minus_p = log(sub(nb.ones_like(predictions_clamped), predictions_clamped))

    # Compute binary cross-entropy
    bce_per_sample = -(
        mul(targets, log_p) + mul(sub(nb.ones_like(targets), targets), log_one_minus_p)
    )

    # Average over batch
    return mean(bce_per_sample)


def softmax_cross_entropy_loss(
    logits: nb.Tensor, targets: nb.Tensor, axis: int = -1
) -> nb.Tensor:
    """Compute softmax cross-entropy loss (numerically stable).

    This is equivalent to cross_entropy_loss but more numerically stable
    by combining softmax and cross-entropy computations.

    Args:
        logits: Raw model outputs [batch_size, num_classes]
        targets: One-hot encoded targets [batch_size, num_classes]
        axis: Axis along which to compute softmax

    Returns:
        Scalar loss value
    """
    from ...ops.binary import mul
    from ...ops.reduce import mean
    from ...ops.reduce import sum as tensor_sum
    from ...ops.special import logsumexp

    # Compute log_softmax = logits - logsumexp(logits)
    log_sum_exp = logsumexp(logits, axis=axis, keep_dims=True)
    log_softmax = logits - log_sum_exp

    # Cross-entropy with log_softmax
    cross_entropy = -tensor_sum(mul(targets, log_softmax), axes=axis)

    # Average over batch
    return mean(cross_entropy)


__all__ = [
    "cross_entropy_loss",
    "sparse_cross_entropy_loss",
    "binary_cross_entropy_loss",
    "softmax_cross_entropy_loss",
]
