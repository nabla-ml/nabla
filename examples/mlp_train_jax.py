#!/usr/bin/env python3
"""JAX implementation to learn the complex 8-period sin curve - exact same logic as Nabla version."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad

# Configuration
BATCH_SIZE = 128
LAYERS = [1, 64, 128, 128, 64, 1]
LEARNING_RATE = 0.01
NUM_EPOCHS = 5000
PRINT_INTERVAL = 200
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


def leaky_relu_manual(x: jnp.ndarray, negative_slope: float = 0.01) -> jnp.ndarray:
    """Manual leaky ReLU implementation matching Nabla version."""
    relu_x = jax.nn.relu(x)
    slope_tensor = jnp.array(negative_slope, dtype=jnp.float32)
    one_minus_slope = jnp.array(1.0 - negative_slope, dtype=jnp.float32)
    return one_minus_slope * relu_x + slope_tensor * x


def mlp_forward_leaky(x: jnp.ndarray, params: list[jnp.ndarray]) -> jnp.ndarray:
    """MLP forward pass with leaky ReLU to prevent dead neurons."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = jnp.matmul(output, w) + b
        # Apply leaky ReLU to all layers except the last
        if i < len(params) - 2:
            output = leaky_relu_manual(output, 0.01)
    return output


def mean_squared_error(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute mean squared error loss."""
    diff = predictions - targets
    squared_errors = diff * diff
    batch_size = jnp.array(predictions.shape[0], dtype=jnp.float32)
    loss = jnp.sum(squared_errors) / batch_size
    return loss


def mlp_forward_and_loss_leaky(params: list[jnp.ndarray], x: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Combined forward pass and loss computation with leaky ReLU."""
    predictions = mlp_forward_leaky(x, params)
    loss = mean_squared_error(predictions, targets)
    return loss


def create_sin_dataset(batch_size: int = 256) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create the COMPLEX 8-period sin dataset."""
    np_x = np.random.uniform(0.0, 1.0, (batch_size, 1)).astype(np.float32)
    np_targets = (np.sin(SIN_PERIODS * 2.0 * np.pi * np_x) / 2.0 + 0.5).astype(
        np.float32
    )

    x = jnp.array(np_x)
    targets = jnp.array(np_targets)
    return x, targets


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
            # Small positive bias to help with leaky ReLU
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
    """AdamW optimizer step with weight decay."""
    updated_params = []
    updated_m = []
    updated_v = []

    # Convert scalars to tensors
    beta1_tensor = jnp.array(beta1, dtype=jnp.float32)
    beta2_tensor = jnp.array(beta2, dtype=jnp.float32)
    lr_tensor = jnp.array(learning_rate, dtype=jnp.float32)
    eps_tensor = jnp.array(eps, dtype=jnp.float32)
    wd_tensor = jnp.array(weight_decay, dtype=jnp.float32)
    one_tensor = jnp.array(1.0, dtype=jnp.float32)

    # Bias correction terms
    beta1_power = jnp.array(beta1**step, dtype=jnp.float32)
    beta2_power = jnp.array(beta2**step, dtype=jnp.float32)
    bias_correction1 = one_tensor - beta1_power
    bias_correction2 = one_tensor - beta2_power

    for param, grad, m, v in zip(params, gradients, m_states, v_states, strict=False):
        # Weight decay (applied to parameters, not gradients)
        param_with_decay = param * (one_tensor - wd_tensor * lr_tensor)

        # Update biased first moment estimate
        new_m = beta1_tensor * m + (one_tensor - beta1_tensor) * grad

        # Update biased second raw moment estimate
        grad_squared = grad * grad
        new_v = beta2_tensor * v + (one_tensor - beta2_tensor) * grad_squared

        # Compute bias-corrected first moment estimate
        m_hat = new_m / bias_correction1

        # Compute bias-corrected second raw moment estimate
        v_hat = new_v / bias_correction2

        # Update parameters
        sqrt_v_hat = v_hat**0.5  # Use ** 0.5 instead of sqrt for consistency
        denominator = sqrt_v_hat + eps_tensor
        update = lr_tensor * m_hat / denominator
        new_param = param_with_decay - update

        updated_params.append(new_param)
        updated_m.append(new_m)
        updated_v.append(new_v)

    return updated_params, updated_m, updated_v


def init_adamw_state(params: list[jnp.ndarray]) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """Initialize AdamW state."""
    m_states = []
    v_states = []
    for param in params:
        m_np = np.zeros_like(param)
        v_np = np.zeros_like(param)
        m_states.append(jnp.array(m_np))
        v_states.append(jnp.array(v_np))
    return m_states, v_states


def learning_rate_schedule(
    epoch: int,
    initial_lr: float = 0.001,
    decay_factor: float = 0.95,
    decay_every: int = 1000,
) -> float:
    """Learning rate schedule for complex function learning."""
    return initial_lr * (decay_factor ** (epoch // decay_every))


def train_step_adamw(
    x: jnp.ndarray,
    targets: jnp.ndarray,
    params: list[jnp.ndarray],
    m_states: list[jnp.ndarray],
    v_states: list[jnp.ndarray],
    step: int,
    learning_rate: float,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray], float]:
    """Perform one training step using AdamW."""
    # Forward pass + gradient computation
    loss_value, gradients = value_and_grad(mlp_forward_and_loss_leaky)(params, x, targets)

    # AdamW optimizer update
    updated_params, updated_m, updated_v = adamw_step(
        params, gradients, m_states, v_states, step, learning_rate
    )

    loss_scalar = float(loss_value)
    return updated_params, updated_m, updated_v, loss_scalar


def analyze_jax_learning_progress(params: list[jnp.ndarray], epoch: int):
    """Analyze how well we're learning the complex function."""
    # Create a dense test set
    x_test_np = np.linspace(0, 1, 1000).reshape(-1, 1).astype(np.float32)
    targets_test_np = (
        np.sin(SIN_PERIODS * 2.0 * np.pi * x_test_np) / 2.0 + 0.5
    ).astype(np.float32)

    x_test = jnp.array(x_test_np)
    targets_test = jnp.array(targets_test_np)

    predictions_test = mlp_forward_leaky(x_test, params)
    test_loss = mean_squared_error(predictions_test, targets_test)

    pred_np = np.array(predictions_test)
    target_np = np.array(targets_test)

    pred_range = pred_np.max() - pred_np.min()
    target_range = target_np.max() - target_np.min()
    range_ratio = pred_range / target_range

    test_loss_scalar = float(test_loss)
    print(f"  Test loss: {test_loss_scalar:.6f}, Range ratio: {range_ratio:.3f}")

    return test_loss_scalar


def test_jax_complex_sin():
    """Test JAX implementation for complex sin learning."""
    print("=== Learning COMPLEX 8-Period Sin Function with JAX ===")
    print(f"Architecture: {LAYERS}")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print(f"Sin periods: {SIN_PERIODS}")
    print(f"Batch size: {BATCH_SIZE}")

    # Initialize for complex function learning
    params = initialize_for_complex_function(LAYERS)
    m_states, v_states = init_adamw_state(params)

    # Initial analysis
    x_init, targets_init = create_sin_dataset(BATCH_SIZE)
    predictions_init = mlp_forward_leaky(x_init, params)
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
    best_test_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Learning rate schedule
        current_lr = learning_rate_schedule(epoch, LEARNING_RATE)

        # Create fresh batch
        x, targets = create_sin_dataset(BATCH_SIZE)

        # Training step
        params, m_states, v_states, loss = train_step_adamw(
            x, targets, params, m_states, v_states, epoch, current_lr
        )

        epoch_time = time.time() - epoch_start_time
        avg_loss += loss
        avg_time += epoch_time

        if epoch % PRINT_INTERVAL == 0:
            print(
                f"\nEpoch {epoch}: Loss = {avg_loss / PRINT_INTERVAL:.6f}, "
                f"LR = {current_lr:.6f}, Time = {avg_time / PRINT_INTERVAL:.4f}s/iter"
            )

            # Detailed analysis
            test_loss = analyze_jax_learning_progress(params, epoch)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                print(f"  New best test loss: {best_test_loss:.6f}")

            avg_loss = 0.0
            avg_time = 0.0

    print("\nJAX training completed!")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    x_test_np = np.linspace(0, 1, 1000).reshape(-1, 1).astype(np.float32)
    targets_test_np = (
        np.sin(SIN_PERIODS * 2.0 * np.pi * x_test_np) / 2.0 + 0.5
    ).astype(np.float32)

    x_test = jnp.array(x_test_np)
    predictions_test = mlp_forward_leaky(x_test, params)

    pred_final_np = np.array(predictions_test)

    final_test_loss = np.mean((pred_final_np - targets_test_np) ** 2)

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
