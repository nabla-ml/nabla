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
"""Multi-Layer Perceptron (MLP) architectures."""

from collections.abc import Callable

import nabla as nb

from ..init.variance_scaling import he_normal, initialize_mlp_params, xavier_normal
from ..layers.linear import mlp_forward_with_activations
from ..losses.regression import mean_squared_error


def create_mlp_forward_and_loss(activation: str = "relu") -> Callable:
    """Create a combined forward pass and loss computation function.

    This function factory creates the forward_and_loss function needed
    for VJP computation in training loops.

    Args:
        activation: Activation function for hidden layers

    Returns:
        Function that takes inputs and returns loss
    """

    def mlp_forward_and_loss(inputs: list[nb.Array]) -> list[nb.Array]:
        """Combined forward pass and loss computation for VJP."""
        x, targets, *params = inputs
        predictions = mlp_forward_with_activations(x, params, activation)
        loss = mean_squared_error(predictions, targets)
        return [loss]

    return mlp_forward_and_loss


def create_mlp_config(
    layers: list[int],
    activation: str = "relu",
    final_activation: str | None = None,
    init_method: str = "he_normal",
    seed: int = 42,
) -> dict:
    """Create MLP configuration dictionary.

    Args:
        layers: List of layer sizes [input, hidden1, hidden2, ..., output]
        activation: Activation function for hidden layers
        final_activation: Optional activation for final layer
        init_method: Weight initialization method
        seed: Random seed for reproducibility

    Returns:
        Configuration dictionary with params and forward function
    """
    # Initialize parameters
    if init_method == "mlp_specialized":
        # Use the specialized initialization from mlp_train_jit.py
        params = initialize_mlp_params(layers, seed)
    elif init_method == "he_normal":
        params = []
        for i in range(len(layers) - 1):
            w = he_normal((layers[i], layers[i + 1]), seed + i)
            b = nb.zeros((1, layers[i + 1]))
            params.extend([w, b])
    elif init_method == "xavier_normal":
        params = []
        for i in range(len(layers) - 1):
            w = xavier_normal((layers[i], layers[i + 1]), seed + i)
            b = nb.zeros((1, layers[i + 1]))
            params.extend([w, b])
    else:
        raise ValueError(f"Unsupported init_method: {init_method}")

    # Create forward function
    def forward_fn(x: nb.Array, params: list[nb.Array]) -> nb.Array:
        return mlp_forward_with_activations(x, params, activation, final_activation)

    # Create forward and loss function for training
    forward_and_loss_fn = create_mlp_forward_and_loss(activation)

    return {
        "params": params,
        "forward": forward_fn,
        "forward_and_loss": forward_and_loss_fn,
        "layers": layers,
        "activation": activation,
        "final_activation": final_activation,
        "init_method": init_method,
    }


class MLPBuilder:
    """Builder class for creating MLP configurations."""

    def __init__(self):
        self.layers = None
        self.activation = "relu"
        self.final_activation = None
        self.init_method = "he_normal"
        self.seed = 42

    def with_layers(self, layers: list[int]) -> "MLPBuilder":
        """Set layer sizes."""
        self.layers = layers
        return self

    def with_activation(self, activation: str) -> "MLPBuilder":
        """Set hidden layer activation function."""
        self.activation = activation
        return self

    def with_final_activation(self, activation: str | None) -> "MLPBuilder":
        """Set final layer activation function."""
        self.final_activation = activation
        return self

    def with_init_method(self, method: str) -> "MLPBuilder":
        """Set weight initialization method."""
        self.init_method = method
        return self

    def with_seed(self, seed: int) -> "MLPBuilder":
        """Set random seed."""
        self.seed = seed
        return self

    def build(self) -> dict:
        """Build the MLP configuration."""
        if self.layers is None:
            raise ValueError("Must specify layers with .with_layers()")

        return create_mlp_config(
            self.layers,
            self.activation,
            self.final_activation,
            self.init_method,
            self.seed,
        )
