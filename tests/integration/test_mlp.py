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

"""Integration test for MLP training using VJP autodiff."""

import numpy as np

import nabla

# Configuration constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_LAYERS = [1, 32, 64, 32, 1]
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_EPOCHS = 30
PRINT_INTERVAL = 10


def mlp_forward(x: nabla.Array, params: list[nabla.Array]) -> nabla.Array:
    """MLP forward pass through all layers.

    Args:
        x: Input data
        params: [w1, b1, w2, b2, w3, b3, ...] - weights and biases

    Returns:
        Network output (predictions)
    """
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nabla.matmul(output, w) + b
        # Apply ReLU to all layers except the last
        if i < len(params) - 2:
            output = nabla.relu(output)
    return output


def mean_squared_error(predictions: nabla.Array, targets: nabla.Array) -> nabla.Array:
    """Compute mean squared error loss.

    Args:
        predictions: Model predictions
        targets: Ground truth targets

    Returns:
        MSE loss
    """
    diff = predictions - targets
    squared_errors = diff * diff
    num_elements = nabla.array(
        [np.float32(predictions.shape[0] * predictions.shape[1])]
    )
    loss = nabla.reduce_sum(squared_errors) / num_elements
    return loss


def mlp_forward_and_loss(inputs: list[nabla.Array]) -> list[nabla.Array]:
    """Combined forward pass and loss computation for VJP.

    Args:
        inputs: [x, targets, w1, b1, w2, b2, w3, b3, ...]

    Returns:
        [loss]
    """
    x, targets, *params = inputs
    predictions = mlp_forward(x, params)
    loss = mean_squared_error(predictions, targets)
    return [loss]


def create_sin_dataset(batch_size: int = 32) -> tuple[nabla.Array, nabla.Array]:
    """Create training data for learning sin function.

    Args:
        batch_size: Number of samples to generate

    Returns:
        Tuple of (inputs, targets) as nabla Arrays
    """
    np_x = np.linspace(0, 1, batch_size, dtype=np.float32).reshape(-1, 1)
    np_targets = (np.sin(3 * np.pi * np_x) + 1) / 2

    x = nabla.Array.from_numpy(np_x)
    targets = nabla.Array.from_numpy(np_targets)

    return x, targets


def initialize_mlp_params(layers: list[int], seed: int = 42) -> list[nabla.Array]:
    """Initialize MLP parameters with Xavier initialization.

    Args:
        layers: List defining network architecture [input_size, hidden1, hidden2, ..., output_size]
        seed: Random seed for reproducible initialization

    Returns:
        List of alternating weight and bias Arrays [w1, b1, w2, b2, ...]
    """
    np.random.seed(seed)
    params = []

    for i in range(len(layers) - 1):
        # Xavier initialization
        fan_in, fan_out = layers[i], layers[i + 1]
        std = np.sqrt(2.0 / (fan_in + fan_out))

        # Create weight and bias arrays
        w_np = np.random.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)
        b_np = np.zeros((1, fan_out), dtype=np.float32)

        w = nabla.Array.from_numpy(w_np)
        b = nabla.Array.from_numpy(b_np)
        params.extend([w, b])

    return params


def train_step(
    x: nabla.Array,
    targets: nabla.Array,
    params: list[nabla.Array],
    learning_rate: float,
) -> tuple[list[nabla.Array], float]:
    """Perform one training step and return updated params and loss.

    Args:
        x: Input data
        targets: Ground truth targets
        params: Current model parameters
        learning_rate: Learning rate for gradient descent

    Returns:
        Tuple of (updated_params, loss_value)
    """
    # Forward pass + VJP for gradients
    all_inputs = [x, targets] + params
    loss_values, vjp_fn = nabla.vjp(mlp_forward_and_loss, all_inputs)

    # Backward pass
    cotangent = [nabla.array([np.float32(1.0)])]
    gradients = vjp_fn(cotangent)

    # Update parameters using gradients
    lr_scalar = nabla.array([np.float32(learning_rate)])
    updated_params = []
    for i in range(len(params)):
        param_idx = i + 2  # Skip x and targets in gradient outputs
        gradient = gradients[param_idx]
        updated_param = params[i] - lr_scalar * gradient
        updated_params.append(updated_param)

    loss_scalar = loss_values[0].to_numpy().item()
    return updated_params, loss_scalar


def test_mlp_training():
    """Test a complete MLP training setup using only VJP."""
    print("=== Testing MLP Training: Learning Sin Function ===")

    # Setup data and model
    x, targets = create_sin_dataset(DEFAULT_BATCH_SIZE)
    params = initialize_mlp_params(DEFAULT_LAYERS)

    print(f"Training data shape: x={x.shape}, targets={targets.shape}")
    print(f"Network architecture: {DEFAULT_LAYERS}")
    print("Starting training...")

    # Training loop
    for epoch in range(DEFAULT_NUM_EPOCHS):
        params, loss = train_step(x, targets, params, DEFAULT_LEARNING_RATE)

        # Print progress every PRINT_INTERVAL epochs
        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss:.6f}")

    print(f"Training completed. Final loss: {loss:.6f}")


if __name__ == "__main__":
    test_mlp_training()
