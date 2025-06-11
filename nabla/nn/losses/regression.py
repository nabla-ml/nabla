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
"""Regression loss functions."""

import numpy as np

import nabla as nb


def mean_squared_error(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute mean squared error loss.

    Args:
        predictions: Predicted values of shape (batch_size, ...)
        targets: Target values of shape (batch_size, ...)

    Returns:
        Scalar loss value
    """
    diff = predictions - targets
    squared_errors = diff * diff
    batch_size = nb.array([np.float32(predictions.shape[0])])
    loss = nb.sum(squared_errors) / batch_size
    return loss


def mean_absolute_error(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute mean absolute error loss.

    Args:
        predictions: Predicted values of shape (batch_size, ...)
        targets: Target values of shape (batch_size, ...)

    Returns:
        Scalar loss value
    """
    diff = predictions - targets
    absolute_errors = nb.abs(diff)
    batch_size = nb.array([np.float32(predictions.shape[0])])
    loss = nb.sum(absolute_errors) / batch_size
    return loss


def huber_loss(
    predictions: nb.Array, targets: nb.Array, delta: float = 1.0
) -> nb.Array:
    """Compute Huber loss (smooth L1 loss).

    Args:
        predictions: Predicted values of shape (batch_size, ...)
        targets: Target values of shape (batch_size, ...)
        delta: Threshold for switching between L1 and L2 loss

    Returns:
        Scalar loss value
    """
    diff = predictions - targets
    abs_diff = nb.abs(diff)

    # Use conditional logic: L2 loss for |diff| <= delta, L1 loss otherwise
    quadratic = 0.5 * diff * diff
    linear = delta * abs_diff - 0.5 * delta * delta

    # Create mask for quadratic vs linear and cast to float for arithmetic
    mask = abs_diff <= delta
    mask_float = nb.cast(mask, quadratic.dtype)
    # Create inverse mask using ones from creation module
    from nabla.ops.creation import ones

    ones_like_mask = ones(mask.shape, dtype=quadratic.dtype)
    inv_mask_float = ones_like_mask - mask_float
    loss_values = mask_float * quadratic + inv_mask_float * linear

    batch_size = nb.array([np.float32(predictions.shape[0])])
    loss = nb.sum(loss_values) / batch_size
    return loss
