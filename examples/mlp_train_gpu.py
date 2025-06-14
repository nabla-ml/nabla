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

import time

import numpy as np
from max import driver

import nabla as nb

# Configuration
BATCH_SIZE = 4
LAYERS = [1, 64, 128, 256, 128, 64, 1]
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
PRINT_INTERVAL = 1
SIN_PERIODS = 8

device = driver.CPU() if driver.accelerator_count() == 0 else driver.Accelerator()
print(f"Using {device} device")


def mlp_forward(x: nb.Array, params: list[nb.Array]) -> nb.Array:
    """MLP forward pass through all layers."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nb.matmul(output, w) + b
        # Apply ReLU to all layers except the last
        if i < len(params) - 2:
            output = nb.relu(output)
    return output


def mean_squared_error(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute mean squared error loss."""
    diff = predictions - targets
    squared_errors = diff * diff
    batch_size = nb.array(predictions.shape[0], dtype=nb.DType.float32).to(device)
    loss = nb.sum(squared_errors) / batch_size
    return loss


def mlp_forward_and_loss(inputs: list[nb.Array]) -> nb.Array:
    """Combined forward pass and loss computation for VJP with leaky ReLU."""
    x, targets, *params = inputs
    predictions = mlp_forward(x, params)
    loss = mean_squared_error(predictions, targets)
    return loss


def create_sin_dataset(batch_size: int = 256) -> tuple[nb.Array, nb.Array]:
    """Create the COMPLEX 8-period sin dataset."""
    x = nb.rand((batch_size, 1), lower=0.0, upper=1.0, dtype=nb.DType.float32).to(
        device
    )
    targets = nb.sin(SIN_PERIODS * 2.0 * np.pi * x) / 2.0 + 0.5
    return x, targets


def initialize_for_complex_function(
    layers: list[int], seed: int = 42
) -> list[nb.Array]:
    """Initialize specifically for learning complex high-frequency functions."""
    np.random.seed(seed)
    params = []

    for i in range(len(layers) - 1):
        fan_in, fan_out = layers[i], layers[i + 1]
        w = nb.he_normal((fan_in, fan_out), seed=seed).to(device)
        b = nb.zeros((fan_out,)).to(device)
        params.append(w)
        params.append(b)

    return params


def adamw_step(
    params: list[nb.Array],
    gradients: list[nb.Array],
    m_states: list[nb.Array],
    v_states: list[nb.Array],
    step: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[nb.Array], list[nb.Array], list[nb.Array]]:
    """AdamW optimizer step with weight decay - OPTIMIZED to match JAX efficiency."""
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


def init_adamw_state(params: list[nb.Array]) -> tuple[list[nb.Array], list[nb.Array]]:
    """Initialize AdamW state - optimized version."""
    m_states = []
    v_states = []
    for param in params:
        # Use zeros_like for more efficient initialization
        m_np = np.zeros_like(param.to_numpy())
        v_np = np.zeros_like(param.to_numpy())
        m_states.append(nb.Array.from_numpy(m_np).to(device))
        v_states.append(nb.Array.from_numpy(v_np).to(device))
    return m_states, v_states


def learning_rate_schedule(
    epoch: int,
    initial_lr: float = 0.001,
    decay_factor: float = 0.95,
    decay_every: int = 1000,
) -> float:
    """Learning rate schedule for complex function learning."""
    return initial_lr * (decay_factor ** (epoch // decay_every))


@nb.jit(show_graph=True)
def train_step(
    x: nb.Array,
    targets: nb.Array,
    params: list[nb.Array],
    m_states: list[nb.Array],
    v_states: list[nb.Array],
    step: int,
    learning_rate: float,
) -> tuple[list[nb.Array], list[nb.Array], list[nb.Array], nb.Array]:
    """JIT-compiled training step combining gradient computation and optimizer update."""

    # Define loss function that takes separate arguments (JAX style)
    def loss_fn(*inner_params):
        predictions = mlp_forward(x, inner_params)
        loss = mean_squared_error(predictions, targets)
        return loss

    loss_value, param_gradients = nb.value_and_grad(
        loss_fn, argnums=list(range(len(params)))
    )(*params)

    # AdamW optimizer update
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, learning_rate
    )

    return updated_params, updated_m, updated_v, loss_value


@nb.jit
def compute_predictions_and_loss(
    x_test: nb.Array, targets_test: nb.Array, params: list[nb.Array]
) -> tuple[nb.Array, nb.Array]:
    """JIT-compiled function to compute predictions and loss."""
    predictions_test = mlp_forward(x_test, params)
    test_loss = mean_squared_error(predictions_test, targets_test)
    return predictions_test, test_loss


def test_nabla_complex_sin():
    """Test Nabla implementation with JIT for complex sin learning."""
    print("=== Learning COMPLEX 8-Period Sin Function with Nabla JIT ===")
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

    pred_init_np = predictions_init.to_numpy()
    target_init_np = targets_init.to_numpy()

    print(f"Initial loss: {initial_loss.to_numpy().item():.6f}")
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

        # Training step using JIT-compiled function
        vjp_start = time.time()

        # Use JIT-compiled training step (combines gradient computation and optimizer update)
        updated_params, updated_m, updated_v, loss_values = train_step(
            x, targets, params, m_states, v_states, epoch, current_lr
        )

        vjp_time = time.time() - vjp_start

        # Update return values (no separate AdamW step needed)
        params, m_states, v_states = updated_params, updated_m, updated_v
        adamw_time = 0.0  # Already included in the JIT step

        # Loss extraction and conversion
        loss_value = loss_values.to_numpy().item()

        epoch_time = time.time() - epoch_start_time
        avg_loss += loss_value
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

    print("\nNabla JIT training completed!")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    x_test_np = np.linspace(0, 1, 1000).reshape(-1, 1).astype(np.float32)
    targets_test_np = (
        np.sin(SIN_PERIODS * 2.0 * np.pi * x_test_np) / 2.0 + 0.5
    ).astype(np.float32)

    x_test = nb.Array.from_numpy(x_test_np).to(device)
    targets_test = nb.Array.from_numpy(targets_test_np).to(device)

    # Use JIT-compiled function for evaluation
    predictions_test, test_loss = compute_predictions_and_loss(
        x_test, targets_test, params
    )

    pred_final_np = predictions_test.to_numpy()

    final_test_loss = test_loss.to_numpy().item()

    print(f"Final test loss: {final_test_loss:.6f}")
    print(
        f"Final predictions range: [{pred_final_np.min():.3f}, {pred_final_np.max():.3f}]"
    )
    print(f"Target range: [{targets_test_np.min():.3f}, {targets_test_np.max():.3f}]")

    # Calculate correlation
    correlation = np.corrcoef(pred_final_np.flatten(), targets_test_np.flatten())[0, 1]
    print(f"Prediction-target correlation: {correlation:.4f}")

    return final_test_loss, correlation


if __name__ == "__main__":
    final_loss, correlation = test_nabla_complex_sin()
    print("\n=== Nabla JIT Summary ===")
    print(f"Final test loss: {final_loss:.6f}")
    print(f"Correlation with true function: {correlation:.4f}")

    if correlation > 0.95:
        print("SUCCESS: Nabla JIT learned the complex function very well! 🎉")
    elif correlation > 0.8:
        print("GOOD: Nabla JIT learned the general shape well! 👍")
    elif correlation > 0.5:
        print("PARTIAL: Some learning but needs improvement 🤔")
    else:
        print("POOR: Nabla JIT failed to learn the complex function 😞")
