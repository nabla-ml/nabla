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

"""Activation functions for neural networks."""

import numpy as np

import nabla as nb


def relu(x: nb.Tensor) -> nb.Tensor:
    """Rectified Linear Unit activation function.

    Args:
        x: Input tensor

    Returns:
        Tensor with ReLU applied element-wise
    """
    return nb.maximum(x, 0)


def leaky_relu(x: nb.Tensor, negative_slope: float = 0.01) -> nb.Tensor:
    """Leaky ReLU activation function.

    Args:
        x: Input tensor
        negative_slope: Slope for negative values

    Returns:
        Tensor with Leaky ReLU applied element-wise
    """
    zeros = nb.zeros_like(x)
    positive_part = nb.maximum(x, zeros)
    negative_part = nb.minimum(x, zeros) * negative_slope
    return positive_part + negative_part


def sigmoid(x: nb.Tensor) -> nb.Tensor:
    """Sigmoid activation function.

    Args:
        x: Input tensor

    Returns:
        Tensor with sigmoid applied element-wise
    """
    # sigmoid(x) = 1 / (1 + exp(-x))
    # For numerical stability, use:
    # sigmoid(x) = exp(x) / (1 + exp(x)) for x >= 0
    # sigmoid(x) = 1 / (1 + exp(-x)) for x < 0

    zeros = nb.zeros_like(x)
    positive_mask = x >= zeros

    # For positive values: exp(x) / (1 + exp(x))
    exp_x = nb.exp(x)
    positive_part = exp_x / (nb.ones_like(x) + exp_x)

    # For negative values: 1 / (1 + exp(-x))
    exp_neg_x = nb.exp(-x)
    negative_part = nb.ones_like(x) / (nb.ones_like(x) + exp_neg_x)

    # Combine using where-like operation
    positive_mask_float = positive_mask.astype(x.dtype)
    negative_mask_float = nb.ones_like(positive_mask_float) - positive_mask_float

    return positive_mask_float * positive_part + negative_mask_float * negative_part


def tanh(x: nb.Tensor) -> nb.Tensor:
    """Hyperbolic tangent activation function.

    Args:
        x: Input tensor

    Returns:
        Tensor with tanh applied element-wise
    """
    return nb.tanh(x)


def gelu(x: nb.Tensor) -> nb.Tensor:
    """Gaussian Error Linear Unit activation function.

    GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution.
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor

    Returns:
        Tensor with GELU applied element-wise
    """
    # Constants for GELU approximation
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    # GELU approximation
    x_cubed = x * x * x
    tanh_input = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_result = tanh(tanh_input)

    half = nb.full_like(x, 0.5)
    one = nb.ones_like(x)

    return half * x * (one + tanh_result)


def swish(x: nb.Tensor, beta: float = 1.0) -> nb.Tensor:
    """Swish (SiLU) activation function.

    Swish(x) = x * sigmoid(β * x)
    When β = 1, this is SiLU (Sigmoid Linear Unit).

    Args:
        x: Input tensor
        beta: Scaling factor for sigmoid

    Returns:
        Tensor with Swish applied element-wise
    """
    scaled_x = x * beta if beta != 1.0 else x

    return x * sigmoid(scaled_x)


def silu(x: nb.Tensor) -> nb.Tensor:
    """Sigmoid Linear Unit (SiLU) activation function.

    SiLU(x) = x * sigmoid(x) = Swish(x, β=1)

    Args:
        x: Input tensor

    Returns:
        Tensor with SiLU applied element-wise
    """
    return swish(x, beta=1.0)


def softmax(x: nb.Tensor, axis: int = -1) -> nb.Tensor:
    """Softmax activation function.

    Args:
        x: Input tensor
        axis: Axis along which to compute softmax

    Returns:
        Tensor with softmax applied along specified axis
    """
    from ...ops.special import softmax as special_softmax

    return special_softmax(x, axis=axis)


def log_softmax(x: nb.Tensor, axis: int = -1) -> nb.Tensor:
    """Log-softmax activation function.

    Args:
        x: Input tensor
        axis: Axis along which to compute log-softmax

    Returns:
        Tensor with log-softmax applied along specified axis
    """
    from ...ops.special import logsumexp

    log_sum_exp = logsumexp(x, axis=axis, keep_dims=True)
    return x - log_sum_exp


# Activation function registry for easy lookup
ACTIVATION_FUNCTIONS = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "gelu": gelu,
    "swish": swish,
    "silu": silu,
    "softmax": softmax,
    "log_softmax": log_softmax,
}


def get_activation(name: str):
    """Get activation function by name.

    Args:
        name: Name of the activation function

    Returns:
        Activation function

    Raises:
        ValueError: If activation function is not found
    """
    if name not in ACTIVATION_FUNCTIONS:
        available = ", ".join(ACTIVATION_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown activation function '{name}'. Available: {available}"
        )

    return ACTIVATION_FUNCTIONS[name]


__all__ = [
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "gelu",
    "swish",
    "silu",
    "softmax",
    "log_softmax",
    "get_activation",
    "ACTIVATION_FUNCTIONS",
]
