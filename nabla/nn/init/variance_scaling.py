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
"""Variance scaling parameter initialization methods."""

import numpy as np
import nabla as nb


def he_normal(shape: tuple[int, ...], seed: int | None = None) -> nb.Array:
    """He normal initialization for ReLU networks.
    
    Uses normal distribution with std = sqrt(2/fan_in) which is optimal
    for ReLU activations.
    
    Args:
        shape: Shape of the parameter tensor
        seed: Random seed for reproducibility
        
    Returns:
        Initialized parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    
    fan_in = shape[0] if len(shape) >= 2 else shape[0]
    std = (2.0 / fan_in) ** 0.5
    
    weights = np.random.normal(0.0, std, shape).astype(np.float32)
    return nb.Array.from_numpy(weights)


def he_uniform(shape: tuple[int, ...], seed: int | None = None) -> nb.Array:
    """He uniform initialization for ReLU networks.
    
    Uses uniform distribution with bound = sqrt(6/fan_in) which is optimal
    for ReLU activations.
    
    Args:
        shape: Shape of the parameter tensor  
        seed: Random seed for reproducibility
        
    Returns:
        Initialized parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    
    fan_in = shape[0] if len(shape) >= 2 else shape[0]
    bound = (6.0 / fan_in) ** 0.5
    
    weights = np.random.uniform(-bound, bound, shape).astype(np.float32)
    return nb.Array.from_numpy(weights)


def xavier_normal(shape: tuple[int, ...], seed: int | None = None) -> nb.Array:
    """Xavier/Glorot normal initialization.
    
    Uses normal distribution with std = sqrt(2/(fan_in + fan_out)) which
    is optimal for sigmoid/tanh activations.
    
    Args:
        shape: Shape of the parameter tensor
        seed: Random seed for reproducibility
        
    Returns:
        Initialized parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    
    if len(shape) >= 2:
        fan_in, fan_out = shape[0], shape[1]
    else:
        fan_in = fan_out = shape[0]
    
    std = (2.0 / (fan_in + fan_out)) ** 0.5
    
    weights = np.random.normal(0.0, std, shape).astype(np.float32)
    return nb.Array.from_numpy(weights)


def xavier_uniform(shape: tuple[int, ...], seed: int | None = None) -> nb.Array:
    """Xavier/Glorot uniform initialization.
    
    Uses uniform distribution with bound = sqrt(6/(fan_in + fan_out)) which
    is optimal for sigmoid/tanh activations.
    
    Args:
        shape: Shape of the parameter tensor
        seed: Random seed for reproducibility
        
    Returns:
        Initialized parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    
    if len(shape) >= 2:
        fan_in, fan_out = shape[0], shape[1]
    else:
        fan_in = fan_out = shape[0]
    
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    
    weights = np.random.uniform(-bound, bound, shape).astype(np.float32)
    return nb.Array.from_numpy(weights)


def lecun_normal(shape: tuple[int, ...], seed: int | None = None) -> nb.Array:
    """LeCun normal initialization.
    
    Uses normal distribution with std = sqrt(1/fan_in) which is optimal
    for SELU activations.
    
    Args:
        shape: Shape of the parameter tensor
        seed: Random seed for reproducibility
        
    Returns:
        Initialized parameter array
    """
    if seed is not None:
        np.random.seed(seed)
    
    fan_in = shape[0] if len(shape) >= 2 else shape[0]
    std = (1.0 / fan_in) ** 0.5
    
    weights = np.random.normal(0.0, std, shape).astype(np.float32)
    return nb.Array.from_numpy(weights)


def initialize_mlp_params(layers: list[int], seed: int = 42) -> list[nb.Array]:
    """Initialize MLP parameters with specialized strategy for complex functions.
    
    This is the original initialization strategy from mlp_train_jit.py,
    optimized for learning high-frequency functions.
    
    Args:
        layers: List of layer sizes [input, hidden1, hidden2, ..., output]
        seed: Random seed for reproducibility
        
    Returns:
        List of parameter arrays [W1, b1, W2, b2, ...]
    """
    np.random.seed(seed)
    params = []

    for i in range(len(layers) - 1):
        fan_in, fan_out = layers[i], layers[i + 1]

        if i == 0:  # First layer - needs to capture high frequency
            # Larger weights for first layer to capture high frequency patterns
            std = (4.0 / fan_in) ** 0.5
        elif i == len(layers) - 2:  # Output layer
            # Conservative output layer
            std = (0.5 / fan_in) ** 0.5
        else:  # Hidden layers
            # Standard He initialization
            std = (2.0 / fan_in) ** 0.5

        w_np = np.random.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

        # Bias initialization strategy
        if i < len(layers) - 2:  # Hidden layers
            # Small positive bias to help with ReLU
            b_np = np.ones((1, fan_out), dtype=np.float32) * 0.05
        else:  # Output layer
            # Initialize output bias to middle of target range
            b_np = np.ones((1, fan_out), dtype=np.float32) * 0.5

        w = nb.Array.from_numpy(w_np)
        b = nb.Array.from_numpy(b_np)
        params.extend([w, b])

    return params