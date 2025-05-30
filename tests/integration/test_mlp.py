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
DEFAULT_BATCH_SIZE = 64
DEFAULT_LAYERS = [1, 64, 128, 128, 64, 1]
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 10000
PRINT_INTERVAL = 100
SIN_PERIODS = 15  # Number of sine wave periods to learn

# Adam optimizer constants
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8


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
    np_targets = (np.sin(SIN_PERIODS * np.pi * np_x) + 1) / 2

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


class AdamOptimizer:
    """Adam optimizer implementation using nabla arrays.

    The Adam optimizer combines the benefits of AdaGrad and RMSProp by computing
    adaptive learning rates for each parameter based on estimates of first and
    second moments of the gradients.
    """

    def __init__(
        self,
        params: list[nabla.Array],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """Initialize Adam optimizer.

        Args:
            params: List of model parameters to optimize
            learning_rate: Step size for parameter updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0

        # Initialize first and second moment estimates
        self.m = []  # First moment (momentum)
        self.v = []  # Second moment (adaptive learning rate)

        for param in params:
            # Initialize moments as zero arrays with same shape as parameters
            m_np = np.zeros_like(param.to_numpy())
            v_np = np.zeros_like(param.to_numpy())
            self.m.append(nabla.Array.from_numpy(m_np))
            self.v.append(nabla.Array.from_numpy(v_np))

    def update(
        self, params: list[nabla.Array], gradients: list[nabla.Array]
    ) -> list[nabla.Array]:
        """Update parameters using Adam optimizer.

        Args:
            params: Current parameters
            gradients: Gradients with respect to parameters

        Returns:
            Updated parameters
        """
        self.step_count += 1
        updated_params = []

        # Convert hyperparameters to nabla Arrays
        beta1_arr = nabla.array([np.float32(self.beta1)])
        beta2_arr = nabla.array([np.float32(self.beta2)])
        eps_arr = nabla.array([np.float32(self.epsilon)])
        lr_arr = nabla.array([np.float32(self.learning_rate)])
        one_arr = nabla.array([np.float32(1.0)])

        # Bias correction terms
        beta1_t = np.float32(self.beta1**self.step_count)
        beta2_t = np.float32(self.beta2**self.step_count)
        lr_corrected = self.learning_rate * np.sqrt(1.0 - beta2_t) / (1.0 - beta1_t)
        lr_corrected_arr = nabla.array([lr_corrected])

        for i in range(len(params)):
            grad = gradients[i]

            # Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
            self.m[i] = beta1_arr * self.m[i] + (one_arr - beta1_arr) * grad

            # Update biased second moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
            self.v[i] = beta2_arr * self.v[i] + (one_arr - beta2_arr) * grad * grad

            # Compute the update using bias-corrected step size
            # θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
            denominator = nabla.power(self.v[i] + eps_arr, 0.5)
            updated_param = params[i] - lr_corrected_arr * self.m[i] / denominator
            updated_params.append(updated_param)

        return updated_params


def train_step_adam(
    x: nabla.Array,
    targets: nabla.Array,
    params: list[nabla.Array],
    optimizer: AdamOptimizer,
) -> tuple[list[nabla.Array], float]:
    """Perform one training step using Adam optimizer and return updated params and loss.

    Args:
        x: Input data
        targets: Ground truth targets
        params: Current model parameters
        optimizer: Adam optimizer instance

    Returns:
        Tuple of (updated_params, loss_value)
    """
    # Forward pass + VJP for gradients
    all_inputs = [x, targets] + params
    loss_values, vjp_fn = nabla.vjp(mlp_forward_and_loss, all_inputs)

    # Backward pass
    cotangent = [nabla.array([np.float32(1.0)])]
    gradients = vjp_fn(cotangent)

    # Extract parameter gradients (skip x and targets)
    param_gradients = gradients[2:]

    # Update parameters using Adam optimizer
    updated_params = optimizer.update(params, param_gradients)

    loss_scalar = loss_values[0].to_numpy().item()
    return updated_params, loss_scalar


def test_mlp_training():
    """Test a complete MLP training setup using Adam optimizer."""
    print("=== Testing MLP Training with Adam Optimizer: Learning Sin Function ===")

    # Setup data and model
    x, targets = create_sin_dataset(DEFAULT_BATCH_SIZE)
    params = initialize_mlp_params(DEFAULT_LAYERS)

    # Initialize Adam optimizer
    optimizer = AdamOptimizer(
        params=params,
        learning_rate=DEFAULT_LEARNING_RATE,
        beta1=DEFAULT_BETA1,
        beta2=DEFAULT_BETA2,
        epsilon=DEFAULT_EPSILON,
    )

    print(f"Training data shape: x={x.shape}, targets={targets.shape}")
    print(f"Network architecture: {DEFAULT_LAYERS}")
    print(
        f"Optimizer: Adam (lr={DEFAULT_LEARNING_RATE}, β1={DEFAULT_BETA1}, β2={DEFAULT_BETA2}, ε={DEFAULT_EPSILON})"
    )
    print("Starting training...")

    # Training loop
    for epoch in range(DEFAULT_NUM_EPOCHS):
        params, loss = train_step_adam(x, targets, params, optimizer)

        # Print progress every PRINT_INTERVAL epochs
        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss:.6f}")

    print(f"Training completed. Final loss: {loss:.6f}")


if __name__ == "__main__":
    # Run Adam optimizer by default
    test_mlp_training()
