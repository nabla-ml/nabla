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

"""JAX implementation to learn the 8-Period sin curve for comparison with Nabla."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

# Configuration
BATCH_SIZE = 128
LAYERS = [1, 64, 128, 256, 128, 64, 1]
LEARNING_RATE = 0.001  # Match Nabla version for fair comparison
NUM_EPOCHS = 1000
PRINT_INTERVAL = 100
SIN_PERIODS = 8


def mlp_forward(x: jnp.ndarray, params: list[jnp.ndarray]) -> jnp.ndarray:
    """MLP forward pass through all layers."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = jnp.matmul(output, w) + b
        # Apply ReLU to all layers except the last
        if i < len(params) - 2:
            output = jax.nn.relu(output)
    return output


def mean_squared_error(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute mean squared error loss."""
    diff = predictions - targets
    squared_errors = diff * diff
    batch_size = jnp.array(predictions.shape[0], dtype=jnp.float32)
    loss = jnp.sum(squared_errors) / batch_size
    return loss


def mlp_forward_and_loss(inputs: list[jnp.ndarray]) -> jnp.ndarray:
    """Combined forward pass and loss computation for VJP with leaky ReLU."""
    x, targets, *params = inputs
    predictions = mlp_forward(x, params)
    loss = mean_squared_error(predictions, targets)
    return loss


def create_sin_dataset(
    batch_size: int = 256, key: jax.Array | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create the 8-Period sin dataset."""
    if key is None:
        # Use numpy for compatibility with original, then convert to JAX
        np_x = np.random.uniform(0.0, 1.0, (batch_size, 1)).astype(np.float32)
        np_x = jnp.array(np_x)  # Convert to JAX array
    else:
        # Use JAX random for reproducibility
        np_x = jax.random.uniform(key, (batch_size, 1), minval=0.0, maxval=1.0)

    np_targets = (jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * np_x) / 2.0 + 0.5).astype(
        jnp.float32
    )

    return np_x, np_targets


def initialize_for_complex_function(
    layers: list[int], seed: int = 42
) -> list[jnp.ndarray]:
    """Initialize specifically for learning complex high-frequency functions."""
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

        w = jnp.array(w_np)
        b = jnp.array(b_np)
        params.extend([w, b])

    return params


def adamw_step(
    params: list[jnp.ndarray],
    gradients: list[jnp.ndarray],
    m_states: list[jnp.ndarray],
    v_states: list[jnp.ndarray],
    step: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray]]:
    """JIT-compiled AdamW optimizer step with weight decay - OPTIMIZED to match JAX efficiency."""
    updated_params = []
    updated_m = []
    updated_v = []

    for param, grad, m, v in zip(params, gradients, m_states, v_states, strict=False):
        # Update moments
        new_m = beta1 * m + (1.0 - beta1) * grad
        new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

        # Bias correction
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        # Corrected moments
        m_corrected = new_m / bias_correction1
        v_corrected = new_v / bias_correction2

        # Parameter update with weight decay
        new_param = param - learning_rate * (
            m_corrected / (v_corrected**0.5 + eps) + weight_decay * param
        )

        # Append updated values
        updated_params.append(new_param)
        updated_m.append(new_m)
        updated_v.append(new_v)

    return updated_params, updated_m, updated_v


def adamw_step_optimized(
    params: list[jnp.ndarray],
    gradients: list[jnp.ndarray],
    m_states: list[jnp.ndarray],
    v_states: list[jnp.ndarray],
    step: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray]]:
    """JIT-compiled AdamW optimizer step with weight decay - OPTIMIZED to match JAX efficiency."""
    updated_params = []
    updated_m = []
    updated_v = []

    for param, grad, m, v in zip(params, gradients, m_states, v_states, strict=False):
        # Update moments
        new_m = beta1 * m + (1.0 - beta1) * grad
        new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

        # Completely fused parameter update - eliminates ALL intermediate variables
        new_param = param * (1.0 - weight_decay * learning_rate) - learning_rate * (
            new_m / (1.0 - beta1**step)
        ) / (((new_v / (1.0 - beta2**step)) ** 0.5) + eps)

        updated_params.append(new_param)
        updated_m.append(new_m)
        updated_v.append(new_v)

    return updated_params, updated_m, updated_v


def init_adamw_state(
    params: list[jnp.ndarray],
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """Initialize AdamW state - optimized version."""
    m_states = []
    v_states = []
    for param in params:
        # Use zeros_like for more efficient initialization
        m_states.append(jnp.zeros_like(param))
        v_states.append(jnp.zeros_like(param))
    return m_states, v_states


def learning_rate_schedule(
    epoch: int,
    initial_lr: float = 0.001,
    decay_factor: float = 0.95,
    decay_every: int = 1000,
) -> float:
    """Learning rate schedule for complex function learning."""
    return initial_lr * (decay_factor ** (epoch // decay_every))


def complete_training_step(
    x: jnp.ndarray,
    targets: jnp.ndarray,
    params: list[jnp.ndarray],
    m_states: list[jnp.ndarray],
    v_states: list[jnp.ndarray],
    step: int,
    learning_rate: float,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray], jnp.ndarray]:
    """Complete JIT-compiled training step with integrated gradient computation."""

    # Direct gradient computation without passing functions
    def loss_fn(params_inner):
        predictions = mlp_forward(x, params_inner)
        loss = mean_squared_error(predictions, targets)
        return loss

    # Compute loss and gradients directly
    loss_value, param_gradients = jax.value_and_grad(loss_fn)(params)

    # AdamW optimizer update using JIT-compiled step
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, learning_rate
    )

    return updated_params, updated_m, updated_v, loss_value


def train_step_adamw_jitted(
    x: jnp.ndarray,
    targets: jnp.ndarray,
    params: list[jnp.ndarray],
    m_states: list[jnp.ndarray],
    v_states: list[jnp.ndarray],
    step: int,
    learning_rate: float,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray], jnp.ndarray]:
    """JIT-compiled training step using JAX value_and_grad."""
    # Forward pass + gradients using value_and_grad
    loss_value, param_gradients = value_and_grad(mlp_forward_and_loss)(
        params, x, targets
    )

    # AdamW optimizer update using JIT-compiled step
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, learning_rate
    )

    return updated_params, updated_m, updated_v, loss_value


def compute_predictions_and_loss(
    x_test: jnp.ndarray, targets_test: jnp.ndarray, params: list[jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled function to compute predictions and loss."""
    predictions_test = mlp_forward(x_test, params)
    test_loss = mean_squared_error(predictions_test, targets_test)
    return predictions_test, test_loss


def analyze_jax_learning_progress(params: list[jnp.ndarray], epoch: int):
    """Analyze how well we're learning the complex function."""
    # Create a dense test set
    x_test_np = jnp.linspace(0, 1, 1000).reshape(-1, 1).astype(jnp.float32)
    targets_test_np = (
        jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x_test_np) / 2.0 + 0.5
    ).astype(jnp.float32)

    # Use JIT-compiled function for evaluation
    predictions_test, test_loss = compute_predictions_and_loss(
        x_test_np, targets_test_np, params
    )

    test_loss_scalar = float(test_loss)

    return test_loss_scalar


def test_jax_complex_sin():
    """Test JAX implementation for complex sin learning."""
    print("=== Learning 8-Period Sin Function with JAX ===")
    print(f"Architecture: {LAYERS}")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print(f"Sin periods: {SIN_PERIODS}")
    print(f"Batch size: {BATCH_SIZE}")

    # Initialize for complex function learning
    params = initialize_for_complex_function(LAYERS)
    m_states, v_states = init_adamw_state(params)

    # Initial analysis
    x_init, targets_init = create_sin_dataset(BATCH_SIZE)
    predictions_init = mlp_forward(x_init, params)
    initial_loss = mean_squared_error(predictions_init, targets_init)

    pred_init_np = np.array(predictions_init)
    target_init_np = np.array(targets_init)

    print(f"Initial loss: {float(initial_loss):.6f}")
    print(
        f"Initial predictions range: [{pred_init_np.min():.3f}, {pred_init_np.max():.3f}]"
    )
    print(f"Targets range: [{target_init_np.min():.3f}, {target_init_np.max():.3f}]")

    print("\nStarting training...")

    # Training loop
    avg_loss = 0.0
    avg_time = 0.0
    avg_data_time = 0.0
    avg_vjp_time = 0.0
    avg_adamw_time = 0.0
    # best_test_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # Learning rate schedule
        current_lr = learning_rate_schedule(epoch, LEARNING_RATE)

        # Create fresh batch
        data_start = time.time()
        x, targets = create_sin_dataset(BATCH_SIZE)
        data_time = time.time() - data_start

        # Training step using fully JIT-compiled function
        vjp_start = time.time()

        # Use complete JIT-compiled training step
        updated_params, updated_m, updated_v, loss_value = complete_training_step(
            x, targets, params, m_states, v_states, epoch, current_lr
        )

        vjp_time = time.time() - vjp_start

        # Update return values (no separate AdamW step needed)
        params, m_states, v_states = updated_params, updated_m, updated_v
        adamw_time = 0.0  # Already included in the JIT step

        # Loss extraction and conversion
        loss_value_scalar = float(loss_value)

        epoch_time = time.time() - epoch_start_time
        avg_loss += loss_value_scalar
        avg_time += epoch_time
        avg_data_time += data_time
        avg_vjp_time += vjp_time
        avg_adamw_time += adamw_time

        if epoch % PRINT_INTERVAL == 0:
            print(f"\n{'=' * 60}")
            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss / PRINT_INTERVAL:.6f} | Time: {avg_time / PRINT_INTERVAL:.4f}s"
            )
            print(f"{'=' * 60}")
            print(
                f"  ├─ Data Gen:   {avg_data_time / PRINT_INTERVAL:.4f}s ({avg_data_time / avg_time * 100:.1f}%)"
            )
            print(
                f"  └─ JIT Step:   {avg_vjp_time / PRINT_INTERVAL:.4f}s ({avg_vjp_time / avg_time * 100:.1f}%)"
            )

            avg_loss = 0.0
            avg_time = 0.0
            avg_data_time = 0.0
            avg_vjp_time = 0.0
            avg_adamw_time = 0.0

    print("\nJAX training completed!")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    x_test_np = jnp.linspace(0, 1, 1000).reshape(-1, 1).astype(jnp.float32)
    targets_test_np = (
        jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x_test_np) / 2.0 + 0.5
    ).astype(jnp.float32)

    predictions_test = mlp_forward(x_test_np, params)

    pred_final_np = np.array(predictions_test)
    targets_final_np = np.array(targets_test_np)

    final_test_loss = np.mean((pred_final_np - targets_final_np) ** 2)

    print(f"Final test loss: {final_test_loss:.6f}")
    print(
        f"Final predictions range: [{pred_final_np.min():.3f}, {pred_final_np.max():.3f}]"
    )
    print(f"Target range: [{targets_final_np.min():.3f}, {targets_final_np.max():.3f}]")

    # Calculate correlation
    correlation = np.corrcoef(pred_final_np.flatten(), targets_final_np.flatten())[0, 1]
    print(f"Prediction-target correlation: {correlation:.4f}")

    return final_test_loss, correlation


if __name__ == "__main__":
    final_loss, correlation = test_jax_complex_sin()
    print("\n=== JAX Summary ===")
    print(f"Final test loss: {final_loss:.6f}")
    print(f"Correlation with true function: {correlation:.4f}")

    if correlation > 0.95:
        print("SUCCESS: JAX learned the complex function very well! 🎉")
    elif correlation > 0.8:
        print("GOOD: JAX learned the general shape well! 👍")
    elif correlation > 0.5:
        print("PARTIAL: Some learning but needs improvement 🤔")
    else:
        print("POOR: JAX failed to learn the complex function 😞")
