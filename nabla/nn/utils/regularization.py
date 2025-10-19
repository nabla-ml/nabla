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

"""Regularization techniques for neural networks."""

import numpy as np

import nabla as nb


def l1_regularization(params: list[nb.Tensor], weight: float = 0.01) -> nb.Tensor:
    """Compute L1 (Lasso) regularization loss.

    L1 regularization adds a penalty equal to the sum of absolute values of parameters.
    This encourages sparsity in the model parameters.

    Args:
        params: List of parameter tensors (typically weights)
        weight: Regularization strength

    Returns:
        Scalar L1 regularization loss
    """
    l1_loss = nb.tensor([0.0])

    for param in params:
        l1_loss = l1_loss + nb.sum(nb.abs(param))

    return weight * l1_loss


def l2_regularization(params: list[nb.Tensor], weight: float = 0.01) -> nb.Tensor:
    """Compute L2 (Ridge) regularization loss.

    L2 regularization adds a penalty equal to the sum of squares of parameters.
    This encourages small parameter values and helps prevent overfitting.

    Args:
        params: List of parameter tensors (typically weights)
        weight: Regularization strength

    Returns:
        Scalar L2 regularization loss
    """
    l2_loss = nb.tensor([0.0])

    for param in params:
        l2_loss = l2_loss + nb.sum(param * param)

    return weight * l2_loss


def elastic_net_regularization(
    params: list[nb.Tensor],
    l1_weight: float = 0.01,
    l2_weight: float = 0.01,
    l1_ratio: float = 0.5,
) -> nb.Tensor:
    """Compute Elastic Net regularization loss.

    Elastic Net combines L1 and L2 regularization:
    ElasticNet = l1_ratio * L1 + (1 - l1_ratio) * L2

    Args:
        params: List of parameter tensors (typically weights)
        l1_weight: L1 regularization strength
        l2_weight: L2 regularization strength
        l1_ratio: Ratio of L1 to L2 regularization (0 = pure L2, 1 = pure L1)

    Returns:
        Scalar Elastic Net regularization loss
    """
    l1_loss = l1_regularization(params, weight=1.0)  # Unweighted
    l2_loss = l2_regularization(params, weight=1.0)  # Unweighted

    combined_loss = (
        l1_ratio * l1_weight * l1_loss + (1 - l1_ratio) * l2_weight * l2_loss
    )

    return combined_loss


def dropout(
    x: nb.Tensor, p: float = 0.5, training: bool = True, seed: int | None = None
) -> nb.Tensor:
    """Apply dropout regularization.

    During training, randomly sets elements to zero with probability p.
    During inference, scales all elements by (1-p) to maintain expected values.

    Args:
        x: Input tensor
        p: Dropout probability (fraction of elements to set to zero)
        training: Whether in training mode (apply dropout) or inference mode
        seed: Random seed for reproducibility

    Returns:
        Tensor with dropout applied
    """
    if not training or p == 0.0:
        return x

    if p >= 1.0:
        return nb.zeros_like(x)

    # Generate random mask
    if seed is not None:
        np.random.seed(seed)

    keep_prob = 1.0 - p
    mask_np = (np.random.random(x.shape) < keep_prob).astype(np.float32)
    mask = nb.Tensor.from_numpy(mask_np)

    # Apply mask and scale
    return (x * mask) / keep_prob


def spectral_normalization(
    weight: nb.Tensor, u: nb.Tensor | None = None, n_iterations: int = 1
) -> tuple[nb.Tensor, nb.Tensor]:
    """Apply spectral normalization to weight matrix.

    Spectral normalization constrains the spectral norm (largest singular value)
    of weight matrices to be at most 1. This stabilizes training of GANs.

    Args:
        weight: Weight matrix to normalize [out_features, in_features]
        u: Left singular vector estimate (updated during training)
        n_iterations: Number of power iterations to approximate largest singular value

    Returns:
        Tuple of (normalized_weight, updated_u)
    """
    weight_shape = weight.shape

    # Reshape weight to 2D if needed
    if len(weight_shape) > 2:
        weight_2d = weight.reshape((weight_shape[0], -1))
    else:
        weight_2d = weight

    out_features, in_features = weight_2d.shape

    # Initialize u if not provided
    if u is None:
        u_np = np.random.normal(0, 1, (out_features,)).astype(np.float32)
        u_init = nb.Tensor.from_numpy(u_np)
    else:
        u_init = u

    # Power iteration to find largest singular value
    for _ in range(n_iterations):
        # v = W^T @ u / ||W^T @ u||
        weight_t = nb.transpose(weight_2d)
        u_reshaped = nb.reshape(u_init, (-1, 1))
        v_temp = nb.matmul(weight_t, u_reshaped)
        v = nb.reshape(v_temp, (-1,))
        v = v / (nb.sqrt(nb.sum(v * v)) + 1e-8)

        # u = W @ v / ||W @ v||
        v_reshaped = nb.reshape(v, (-1, 1))
        u_temp = nb.matmul(weight_2d, v_reshaped)
        u_init = nb.reshape(u_temp, (-1,))
        u_init = u_init / (nb.sqrt(nb.sum(u_init * u_init)) + 1e-8)

    # Compute spectral norm: sigma = u^T @ W @ v
    weight_t = nb.transpose(weight_2d)
    u_reshaped = nb.reshape(u_init, (-1, 1))
    v_temp = nb.matmul(weight_t, u_reshaped)
    v = nb.reshape(v_temp, (-1,))
    v = v / (nb.sqrt(nb.sum(v * v)) + 1e-8)

    v_reshaped = nb.reshape(v, (-1, 1))
    sigma_temp = nb.matmul(weight_2d, v_reshaped)
    sigma_vec = nb.reshape(sigma_temp, (-1,))
    sigma = nb.sum(u_init * sigma_vec)

    # Normalize weight by spectral norm
    normalized_weight_2d = weight_2d / (sigma + 1e-8)

    # Reshape back to original shape
    if len(weight_shape) > 2:
        normalized_weight = normalized_weight_2d.reshape(weight_shape)
    else:
        normalized_weight = normalized_weight_2d

    return normalized_weight, u_init


def gradient_clipping(
    gradients: list[nb.Tensor], max_norm: float = 1.0, norm_type: str = "l2"
) -> tuple[list[nb.Tensor], nb.Tensor]:
    """Apply gradient clipping to prevent exploding gradients.

    Args:
        gradients: List of gradient tensors
        max_norm: Maximum allowed gradient norm
        norm_type: Type of norm to use ("l2" or "l1")

    Returns:
        Tuple of (clipped_gradients, total_norm)
    """
    # Compute total gradient norm
    if norm_type == "l2":
        total_norm_sq = nb.tensor([0.0])
        for grad in gradients:
            total_norm_sq = total_norm_sq + nb.sum(grad * grad)
        total_norm = nb.sqrt(total_norm_sq)
    elif norm_type == "l1":
        total_norm = nb.tensor([0.0])
        for grad in gradients:
            total_norm = total_norm + nb.sum(nb.abs(grad))
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    # Clip gradients if norm exceeds threshold
    max_norm_tensor = nb.tensor([max_norm])
    clip_coeff = nb.minimum(max_norm_tensor / (total_norm + 1e-8), nb.tensor([1.0]))

    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(grad * clip_coeff)

    return clipped_gradients, total_norm


__all__ = [
    "l1_regularization",
    "l2_regularization",
    "elastic_net_regularization",
    "dropout",
    "spectral_normalization",
    "gradient_clipping",
]
