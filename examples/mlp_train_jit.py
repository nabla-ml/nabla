#!/usr/bin/env python3
"""Improved Endia implementation to learn the complex 8-period sin curve."""

import time

import numpy as np

import endia as nd

# Configuration
BATCH_SIZE = 128
LAYERS = [1, 64, 128, 128, 64, 1]
LEARNING_RATE = 0.001
NUM_EPOCHS = 5000
PRINT_INTERVAL = 200
SIN_PERIODS = 8


def mlp_forward(x: nd.Array, params: list[nd.Array]) -> nd.Array:
    """MLP forward pass through all layers."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nd.matmul(output, w) + b
        # Apply ReLU to all layers except the last
        if i < len(params) - 2:
            output = nd.relu(output)
    return output


def leaky_relu_manual(x: nd.Array, negative_slope: float = 0.01) -> nd.Array:
    """Manual leaky ReLU implementation since Endia might not have it."""
    relu_x = nd.relu(x)
    slope_tensor = nd.array([np.float32(negative_slope)])
    one_minus_slope = nd.array([np.float32(1.0 - negative_slope)])
    return one_minus_slope * relu_x + slope_tensor * x


def mlp_forward_leaky(x: nd.Array, params: list[nd.Array]) -> nd.Array:
    """MLP forward pass with leaky ReLU to prevent dead neurons."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nd.matmul(output, w) + b
        # Apply leaky ReLU to all layers except the last
        if i < len(params) - 2:
            output = leaky_relu_manual(output, 0.01)
    return output


def mean_squared_error(predictions: nd.Array, targets: nd.Array) -> nd.Array:
    """Compute mean squared error loss."""
    diff = predictions - targets
    squared_errors = diff * diff
    batch_size = nd.array([np.float32(predictions.shape[0])])
    loss = nd.sum(squared_errors) / batch_size
    return loss


def mlp_forward_and_loss_leaky(inputs: list[nd.Array]) -> list[nd.Array]:
    """Combined forward pass and loss computation for VJP with leaky ReLU."""
    x, targets, *params = inputs
    predictions = mlp_forward_leaky(x, params)
    loss = mean_squared_error(predictions, targets)
    return [loss]


@nd.jit
def create_sin_dataset(batch_size: int = 256) -> tuple[nd.Array, nd.Array]:
    """Create the COMPLEX 8-period sin dataset."""
    np_x = np.random.uniform(0.0, 1.0, (batch_size, 1)).astype(np.float32)
    np_targets = (np.sin(SIN_PERIODS * 2.0 * np.pi * np_x) / 2.0 + 0.5).astype(
        np.float32
    )

    x = nd.Array.from_numpy(np_x)
    targets = nd.Array.from_numpy(np_targets)
    return x, targets


def initialize_for_complex_function(
    layers: list[int], seed: int = 42
) -> list[nd.Array]:
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

        w = nd.Array.from_numpy(w_np)
        b = nd.Array.from_numpy(b_np)
        params.extend([w, b])

    return params


def adamw_step(
    params: list[nd.Array],
    gradients: list[nd.Array],
    m_states: list[nd.Array],
    v_states: list[nd.Array],
    step: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[nd.Array], list[nd.Array], list[nd.Array]]:
    """AdamW optimizer step with weight decay."""
    updated_params = []
    updated_m = []
    updated_v = []

    # Convert scalars to tensors
    beta1_tensor = nd.array([np.float32(beta1)])
    beta2_tensor = nd.array([np.float32(beta2)])
    lr_tensor = nd.array([np.float32(learning_rate)])
    eps_tensor = nd.array([np.float32(eps)])
    wd_tensor = nd.array([np.float32(weight_decay)])
    one_tensor = nd.array([np.float32(1.0)])

    # Bias correction terms
    beta1_power = nd.array([np.float32(beta1**step)])
    beta2_power = nd.array([np.float32(beta2**step)])
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
        # sqrt_v_hat = nd.sqrt(v_hat)
        sqrt_v_hat = v_hat**0.5  # Use ** 0.5 instead of sqrt for Endia compatibility
        denominator = sqrt_v_hat + eps_tensor
        update = lr_tensor * m_hat / denominator
        new_param = param_with_decay - update

        updated_params.append(new_param)
        updated_m.append(new_m)
        updated_v.append(new_v)

    return updated_params, updated_m, updated_v


def init_adamw_state(params: list[nd.Array]) -> tuple[list[nd.Array], list[nd.Array]]:
    """Initialize AdamW state."""
    m_states = []
    v_states = []
    for param in params:
        m_np = np.zeros_like(param.to_numpy())
        v_np = np.zeros_like(param.to_numpy())
        m_states.append(nd.Array.from_numpy(m_np))
        v_states.append(nd.Array.from_numpy(v_np))
    return m_states, v_states


def learning_rate_schedule(
    epoch: int,
    initial_lr: float = 0.001,
    decay_factor: float = 0.95,
    decay_every: int = 1000,
) -> float:
    """Learning rate schedule for complex function learning."""
    return initial_lr * (decay_factor ** (epoch // decay_every))


@nd.jit
def train_step_adamw(
    x: nd.Array,
    targets: nd.Array,
    params: list[nd.Array],
    m_states: list[nd.Array],
    v_states: list[nd.Array],
    step: int,
    learning_rate: float,
) -> tuple[list[nd.Array], list[nd.Array], list[nd.Array], float]:
    """Perform one training step using AdamW."""
    # Forward pass + VJP for gradients
    all_inputs = [x, targets] + params
    loss_values, vjp_fn = nd.vjp(mlp_forward_and_loss_leaky, all_inputs)

    jitted_vjp_fn = vjp_fn

    # Backward pass
    cotangent = [nd.array([np.float32(1.0)])]
    gradients = jitted_vjp_fn(cotangent)[0]

    # Extract parameter gradients (skip x and targets)
    param_gradients = gradients[2:]

    # AdamW optimizer update
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, learning_rate
    )

    loss_scalar = loss_values[0].to_numpy().item()
    return updated_params, updated_m, updated_v, loss_scalar


def analyze_endia_learning_progress(params: list[nd.Array], epoch: int):
    """Analyze how well we're learning the complex function."""
    # Create a dense test set
    x_test_np = np.linspace(0, 1, 1000).reshape(-1, 1).astype(np.float32)
    targets_test_np = (
        np.sin(SIN_PERIODS * 2.0 * np.pi * x_test_np) / 2.0 + 0.5
    ).astype(np.float32)

    x_test = nd.Array.from_numpy(x_test_np)
    targets_test = nd.Array.from_numpy(targets_test_np)

    predictions_test = mlp_forward_leaky(x_test, params)
    test_loss = mean_squared_error(predictions_test, targets_test)

    pred_np = predictions_test.to_numpy()
    target_np = targets_test.to_numpy()

    pred_range = pred_np.max() - pred_np.min()
    target_range = target_np.max() - target_np.min()
    range_ratio = pred_range / target_range

    test_loss_scalar = test_loss.to_numpy().item()
    print(f"  Test loss: {test_loss_scalar:.6f}, Range ratio: {range_ratio:.3f}")

    return test_loss_scalar


def test_endia_complex_sin():
    """Test Endia implementation for complex sin learning."""
    print("=== Learning COMPLEX 8-Period Sin Function with Endia ===")
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
    best_test_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # Learning rate schedule
        current_lr = learning_rate_schedule(epoch, LEARNING_RATE)

        # Create fresh batch using jitted function with decorator syntax
        x, targets = create_sin_dataset(BATCH_SIZE)

        # Training step using jitted function with decorator syntax
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
            test_loss = analyze_endia_learning_progress(params, epoch)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                print(f"  New best test loss: {best_test_loss:.6f}")

            avg_loss = 0.0
            avg_time = 0.0

    print("\nEndia training completed!")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    x_test_np = np.linspace(0, 1, 1000).reshape(-1, 1).astype(np.float32)
    targets_test_np = (
        np.sin(SIN_PERIODS * 2.0 * np.pi * x_test_np) / 2.0 + 0.5
    ).astype(np.float32)

    x_test = nd.Array.from_numpy(x_test_np)
    predictions_test = mlp_forward_leaky(x_test, params)

    pred_final_np = predictions_test.to_numpy()

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
    final_loss, correlation = test_endia_complex_sin()
    print("\n=== Endia Summary ===")
    print(f"Final test loss: {final_loss:.6f}")
    print(f"Correlation with true function: {correlation:.4f}")

    if correlation > 0.95:
        print("SUCCESS: Endia learned the complex function very well! 🎉")
    elif correlation > 0.8:
        print("GOOD: Endia learned the general shape well! 👍")
    elif correlation > 0.5:
        print("PARTIAL: Some learning but needs improvement 🤔")
    else:
        print("POOR: Endia failed to learn the complex function 😞")
