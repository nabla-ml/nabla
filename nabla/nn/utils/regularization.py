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


def l1_regularization(params: list[nb.Array], weight: float = 0.01) -> nb.Array:
    """Compute L1 (Lasso) regularization loss.
    
    L1 regularization adds a penalty equal to the sum of absolute values of parameters.
    This encourages sparsity in the model parameters.
    
    Args:
        params: List of parameter arrays (typically weights)
        weight: Regularization strength
        
    Returns:
        Scalar L1 regularization loss
    """
    l1_loss = nb.array([0.0])
    
    for param in params:
        l1_loss = l1_loss + nb.sum(nb.abs(param))
    
    return weight * l1_loss


def l2_regularization(params: list[nb.Array], weight: float = 0.01) -> nb.Array:
    """Compute L2 (Ridge) regularization loss.
    
    L2 regularization adds a penalty equal to the sum of squares of parameters.
    This encourages small parameter values and helps prevent overfitting.
    
    Args:
        params: List of parameter arrays (typically weights)
        weight: Regularization strength
        
    Returns:
        Scalar L2 regularization loss
    """
    l2_loss = nb.array([0.0])
    
    for param in params:
        l2_loss = l2_loss + nb.sum(param * param)
    
    return weight * l2_loss


def elastic_net_regularization(
    params: list[nb.Array], 
    l1_weight: float = 0.01, 
    l2_weight: float = 0.01,
    l1_ratio: float = 0.5
) -> nb.Array:
    """Compute Elastic Net regularization loss.
    
    Elastic Net combines L1 and L2 regularization:
    ElasticNet = l1_ratio * L1 + (1 - l1_ratio) * L2
    
    Args:
        params: List of parameter arrays (typically weights)
        l1_weight: L1 regularization strength
        l2_weight: L2 regularization strength  
        l1_ratio: Ratio of L1 to L2 regularization (0 = pure L2, 1 = pure L1)
        
    Returns:
        Scalar Elastic Net regularization loss
    """
    l1_loss = l1_regularization(params, weight=1.0)  # Unweighted
    l2_loss = l2_regularization(params, weight=1.0)  # Unweighted
    
    total_weight = l1_weight + l2_weight
    combined_loss = l1_ratio * l1_weight * l1_loss + (1 - l1_ratio) * l2_weight * l2_loss
    
    return combined_loss


def dropout(x: nb.Array, p: float = 0.5, training: bool = True, seed: int | None = None) -> nb.Array:
    """Apply dropout regularization.
    
    During training, randomly sets elements to zero with probability p.
    During inference, scales all elements by (1-p) to maintain expected values.
    
    Args:
        x: Input array
        p: Dropout probability (fraction of elements to set to zero)
        training: Whether in training mode (apply dropout) or inference mode
        seed: Random seed for reproducibility
        
    Returns:
        Array with dropout applied
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
    mask = nb.Array.from_numpy(mask_np)
    
    # Apply mask and scale
    return (x * mask) / keep_prob


def spectral_normalization(weight: nb.Array, u: nb.Array | None = None, n_iterations: int = 1) -> tuple[nb.Array, nb.Array]:
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
        u = nb.Array.from_numpy(u_np)
    
    # Power iteration to find largest singular value
    for _ in range(n_iterations):
        # v = W^T @ u / ||W^T @ u||
        v = nb.matmul(weight_2d.T, u.reshape((-1, 1))).reshape((-1,))
        v = v / (nb.sqrt(nb.sum(v * v)) + 1e-8)
        
        # u = W @ v / ||W @ v||  
        u = nb.matmul(weight_2d, v.reshape((-1, 1))).reshape((-1,))
        u = u / (nb.sqrt(nb.sum(u * u)) + 1e-8)
    
    # Compute spectral norm: sigma = u^T @ W @ v
    v = nb.matmul(weight_2d.T, u.reshape((-1, 1))).reshape((-1,))
    v = v / (nb.sqrt(nb.sum(v * v)) + 1e-8)
    sigma = nb.sum(u * nb.matmul(weight_2d, v.reshape((-1, 1))).reshape((-1,)))
    
    # Normalize weight by spectral norm
    normalized_weight_2d = weight_2d / (sigma + 1e-8)
    
    # Reshape back to original shape
    if len(weight_shape) > 2:
        normalized_weight = normalized_weight_2d.reshape(weight_shape)
    else:
        normalized_weight = normalized_weight_2d
    
    return normalized_weight, u


def gradient_clipping(
    gradients: list[nb.Array], 
    max_norm: float = 1.0, 
    norm_type: str = "l2"
) -> tuple[list[nb.Array], nb.Array]:
    """Apply gradient clipping to prevent exploding gradients.
    
    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed gradient norm
        norm_type: Type of norm to use ("l2" or "l1")
        
    Returns:
        Tuple of (clipped_gradients, total_norm)
    """
    # Compute total gradient norm
    if norm_type == "l2":
        total_norm_sq = nb.array([0.0])
        for grad in gradients:
            total_norm_sq = total_norm_sq + nb.sum(grad * grad)
        total_norm = nb.sqrt(total_norm_sq)
    elif norm_type == "l1":
        total_norm = nb.array([0.0])
        for grad in gradients:
            total_norm = total_norm + nb.sum(nb.abs(grad))
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
    
    # Clip gradients if norm exceeds threshold
    max_norm_tensor = nb.array([max_norm])
    clip_coeff = nb.minimum(max_norm_tensor / (total_norm + 1e-8), nb.array([1.0]))
    
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
