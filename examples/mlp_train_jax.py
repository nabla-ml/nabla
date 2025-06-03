#!/usr/bin/env python3
"""JAX implementation to learn the complex 8-period sin curve (equivalent to Nabla version)."""

import time

import jax
import jax.numpy as jnp
import numpy as np

# Configuration - matches the Nabla version
BATCH_SIZE = 128
LAYERS = [1, 64, 128, 128, 64, 1]
LEARNING_RATE = 0.001
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


def leaky_relu(x: jnp.ndarray, negative_slope: float = 0.01) -> jnp.ndarray:
    """Leaky ReLU implementation."""
    return jnp.where(x > 0, x, negative_slope * x)


def mlp_forward_leaky(x: jnp.ndarray, params: list[jnp.ndarray]) -> jnp.ndarray:
    """MLP forward pass with leaky ReLU to prevent dead neurons."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = jnp.matmul(output, w) + b
        # Apply leaky ReLU to all layers except the last
        if i < len(params) - 2:
            output = leaky_relu(output, 0.01)
    return output


# Make a specific fixed-layers implementation for our architecture
@jax.jit
def mlp_forward_leaky_fixed(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    """MLP forward pass with 5 layers (hardcoded for efficiency)"""
    # Layer 1
    output = jnp.matmul(x, w1) + b1
    output = leaky_relu(output, 0.01)

    # Layer 2
    output = jnp.matmul(output, w2) + b2
    output = leaky_relu(output, 0.01)

    # Layer 3
    output = jnp.matmul(output, w3) + b3
    output = leaky_relu(output, 0.01)

    # Layer 4
    output = jnp.matmul(output, w4) + b4
    output = leaky_relu(output, 0.01)

    # Output layer
    output = jnp.matmul(output, w5) + b5

    return output


def mean_squared_error(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute mean squared error loss."""
    diff = predictions - targets
    squared_errors = diff * diff
    batch_size = predictions.shape[0]
    loss = jnp.sum(squared_errors) / batch_size
    return loss


def mlp_forward_and_loss_leaky(inputs: list[jnp.ndarray]) -> jnp.ndarray:
    """Combined forward pass and loss computation."""
    x, targets, *params = inputs
    predictions = mlp_forward_leaky(x, params)
    loss = mean_squared_error(predictions, targets)
    return loss


def create_sin_dataset(key, batch_size=256):
    """Create the COMPLEX 8-period sin dataset."""
    # Generate random x values between 0 and 1
    x = jax.random.uniform(key, (batch_size, 1), dtype=jnp.float32)
    targets = (jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x) / 2.0 + 0.5).astype(jnp.float32)

    return x, targets


def initialize_for_complex_function(
    layers: list[int], seed: int = 42
) -> list[jnp.ndarray]:
    """Initialize specifically for learning complex high-frequency functions."""
    key = jax.random.PRNGKey(seed)
    params = []

    for i in range(len(layers) - 1):
        fan_in, fan_out = layers[i], layers[i + 1]
        key, w_key, b_key = jax.random.split(key, 3)

        if i == 0:  # First layer - needs to capture high frequency
            # Larger weights for first layer to capture high frequency patterns
            std = (4.0 / fan_in) ** 0.5
        elif i == len(layers) - 2:  # Output layer
            # Conservative output layer
            std = (0.5 / fan_in) ** 0.5
        else:  # Hidden layers
            # Standard He initialization
            std = (2.0 / fan_in) ** 0.5

        w = jax.random.normal(w_key, (fan_in, fan_out), dtype=jnp.float32) * std

        # Bias initialization strategy
        if i < len(layers) - 2:  # Hidden layers
            # Small positive bias to help with leaky ReLU
            b = jnp.ones((1, fan_out), dtype=jnp.float32) * 0.05
        else:  # Output layer
            # Initialize output bias to middle of target range
            b = jnp.ones((1, fan_out), dtype=jnp.float32) * 0.5

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

    # Bias correction terms
    beta1_power = beta1**step
    beta2_power = beta2**step
    bias_correction1 = 1.0 - beta1_power
    bias_correction2 = 1.0 - beta2_power

    for param, grad, m, v in zip(params, gradients, m_states, v_states, strict=False):
        # Weight decay (applied to parameters, not gradients)
        param_with_decay = param * (1.0 - weight_decay * learning_rate)

        # Update biased first moment estimate
        new_m = beta1 * m + (1.0 - beta1) * grad

        # Update biased second raw moment estimate
        grad_squared = grad * grad
        new_v = beta2 * v + (1.0 - beta2) * grad_squared

        # Compute bias-corrected first moment estimate
        m_hat = new_m / bias_correction1

        # Compute bias-corrected second raw moment estimate
        v_hat = new_v / bias_correction2

        # Update parameters
        sqrt_v_hat = jnp.sqrt(v_hat)
        denominator = sqrt_v_hat + eps
        update = learning_rate * m_hat / denominator
        new_param = param_with_decay - update

        updated_params.append(new_param)
        updated_m.append(new_m)
        updated_v.append(new_v)

    return updated_params, updated_m, updated_v


def init_adamw_state(
    params: list[jnp.ndarray],
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """Initialize AdamW state."""
    m_states = [jnp.zeros_like(param) for param in params]
    v_states = [jnp.zeros_like(param) for param in params]
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
    all_inputs = [x, targets] + params

    # Compute the loss and gradient
    loss_value, grads = jax.value_and_grad(mlp_forward_and_loss_leaky)(all_inputs)

    # Extract parameter gradients (skip x and targets)
    param_gradients = grads[2:]

    # AdamW optimizer update
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, learning_rate
    )

    return updated_params, updated_m, updated_v, loss_value


# Use JAX's scan for more efficient optimization
@jax.jit
def adamw_update_single_param(carry, inp):
    param, grad, m, v, step, lr, beta1, beta2, eps, wd = inp

    # Weight decay
    param_with_decay = param * (1.0 - wd * lr)

    # Update moments
    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

    # Bias correction
    m_hat = new_m / (1.0 - beta1**step)
    v_hat = new_v / (1.0 - beta2**step)

    # Update parameters
    new_param = param_with_decay - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    return carry, (new_param, new_m, new_v)


# Create JIT-compiled versions with proper static arguments
@jax.jit
def train_step_adamw_jitted(
    x, targets, params, m_states, v_states, step, learning_rate
):
    """JIT-compiled training step."""
    # Forward pass + gradient computation
    all_inputs = [x, targets] + params
    loss_value, grads = jax.value_and_grad(mlp_forward_and_loss_leaky)(all_inputs)
    param_gradients = grads[2:]  # Skip x and targets gradients

    # Parameter update
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 0.01

    # Create updated parameters using list comprehension (better for JAX)
    updated_params = []
    updated_m = []
    updated_v = []

    for param, grad, m, v in zip(
        params, param_gradients, m_states, v_states, strict=False
    ):
        # Weight decay
        param_with_decay = param * (1.0 - weight_decay * learning_rate)

        # Update moments
        new_m = beta1 * m + (1.0 - beta1) * grad
        new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

        # Bias correction
        m_hat = new_m / (1.0 - beta1**step)
        v_hat = new_v / (1.0 - beta2**step)

        # Update parameters
        new_param = param_with_decay - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        updated_params.append(new_param)
        updated_m.append(new_m)
        updated_v.append(new_v)

    return updated_params, updated_m, updated_v, loss_value


@jax.jit
def compute_predictions_and_loss(x_test, targets_test, params):
    """JIT-compiled function to compute predictions and loss."""
    predictions_test = mlp_forward_leaky(x_test, params)
    test_loss = mean_squared_error(predictions_test, targets_test)
    return predictions_test, test_loss


def analyze_jax_learning_progress(params: list[jnp.ndarray], epoch: int):
    """Analyze how well we're learning the complex function."""
    # Create a dense test set
    x_test = jnp.linspace(0, 1, 1000).reshape(-1, 1).astype(jnp.float32)
    targets_test = (jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x_test) / 2.0 + 0.5).astype(
        jnp.float32
    )

    predictions_test, test_loss = compute_predictions_and_loss(
        x_test, targets_test, params
    )

    pred_range = jnp.max(predictions_test) - jnp.min(predictions_test)
    target_range = jnp.max(targets_test) - jnp.min(targets_test)
    range_ratio = pred_range / target_range

    test_loss_scalar = float(test_loss)
    print(f"  Test loss: {test_loss_scalar:.6f}, Range ratio: {float(range_ratio):.3f}")

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

    # Create key for random generation
    key = jax.random.PRNGKey(42)

    # Initial analysis
    key, subkey = jax.random.split(key)
    x_init, targets_init = create_sin_dataset(subkey, BATCH_SIZE)
    predictions_init = mlp_forward_leaky(x_init, params)
    initial_loss = mean_squared_error(predictions_init, targets_init)

    print(f"Initial loss: {float(initial_loss):.6f}")
    print(
        f"Initial predictions range: [{float(jnp.min(predictions_init)):.3f}, {float(jnp.max(predictions_init)):.3f}]"
    )
    print(
        f"Targets range: [{float(jnp.min(targets_init)):.3f}, {float(jnp.max(targets_init)):.3f}]"
    )

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
        key, subkey = jax.random.split(key)
        x, targets = create_sin_dataset(subkey, BATCH_SIZE)

        # Training step using jitted function
        params, m_states, v_states, loss = train_step_adamw_jitted(
            x, targets, params, m_states, v_states, epoch, current_lr
        )

        epoch_time = time.time() - epoch_start_time
        avg_loss += float(loss)
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
    x_test = jnp.linspace(0, 1, 1000).reshape(-1, 1).astype(jnp.float32)
    targets_test = (jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x_test) / 2.0 + 0.5).astype(
        jnp.float32
    )

    predictions_test = mlp_forward_leaky(x_test, params)

    pred_final_np = np.array(predictions_test)
    targets_test_np = np.array(targets_test)

    final_test_loss = float(mean_squared_error(predictions_test, targets_test))

    print(f"Final test loss: {final_test_loss:.6f}")
    print(
        f"Final predictions range: [{float(jnp.min(predictions_test)):.3f}, {float(jnp.max(predictions_test)):.3f}]"
    )
    print(
        f"Target range: [{float(jnp.min(targets_test)):.3f}, {float(jnp.max(targets_test)):.3f}]"
    )

    # Calculate correlation
    correlation = np.corrcoef(pred_final_np.flatten(), targets_test_np.flatten())[0, 1]
    print(f"Prediction-target correlation: {correlation:.4f}")

    return final_test_loss, correlation


if __name__ == "__main__":
    # Enable JAX memory preallocation
    jax.config.update("jax_platform_name", "cpu")  # Change to 'gpu' to use GPU

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
