#!/usr/bin/env python3
"""Nabla implementation with JIT acceleration using modular nn components."""

import time

import numpy as np

import nabla as nb

# Import from the new modular nn structure
from nabla.nn import (
    adamw_step,
    create_sin_dataset,
    init_adamw_state,
    initialize_mlp_params,
    learning_rate_schedule,
    mean_squared_error,
    mlp_forward,
    value_and_grad,
)

# Configuration
BATCH_SIZE = 128
LAYERS = [1, 64, 128, 256, 128, 64, 1]
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
PRINT_INTERVAL = 100
SIN_PERIODS = 8


def mlp_forward_and_loss(inputs: list[nb.Array]) -> nb.Array:
    """Combined forward pass and loss computation for VJP with leaky ReLU."""
    x, targets, *params = inputs
    predictions = mlp_forward(x, params)
    loss = mean_squared_error(predictions, targets)
    return loss


def test_nabla_modular_mlp():
    """Test Nabla implementation with modular nn components."""
    print("=== Learning COMPLEX 8-Period Sin Function with Modular Nabla ===")
    print(f"Architecture: {LAYERS}")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print(f"Sin periods: {SIN_PERIODS}")
    print(f"Batch size: {BATCH_SIZE}")

    # Initialize for complex function learning using modular components
    params = initialize_mlp_params(LAYERS)
    m_states, v_states = init_adamw_state(params)

    # Initial analysis
    x_init, targets_init = create_sin_dataset(BATCH_SIZE, SIN_PERIODS)
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

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # Learning rate schedule using modular component
        current_lr = learning_rate_schedule(epoch, LEARNING_RATE)

        # Create fresh batch using modular component
        data_start = time.time()
        x, targets = create_sin_dataset(BATCH_SIZE, SIN_PERIODS)
        data_time = time.time() - data_start

        # Training step using JIT-compiled function
        vjp_start = time.time()
        all_inputs = [x, targets] + params

        # Use modular value_and_grad to compute loss and gradients
        # Only compute gradients w.r.t. parameters (indices 2 onwards)
        param_indices = list(range(2, 2 + len(params)))
        loss_values, param_gradients = value_and_grad(
            mlp_forward_and_loss, argnums=param_indices
        )(all_inputs)

        vjp_time = time.time() - vjp_start

        # Optimizer step using modular adamw_step
        adamw_start = time.time()
        updated_params, updated_m, updated_v = adamw_step(
            params, param_gradients, m_states, v_states, epoch, current_lr
        )
        adamw_time = time.time() - adamw_start

        # Update return values
        params, m_states, v_states = updated_params, updated_m, updated_v

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
                f"  ├─ VJP Comp:   {avg_vjp_time / PRINT_INTERVAL:.4f}s ({avg_vjp_time / avg_time * 100:.1f}%)"
            )
            print(
                f"  └─ AdamW Step: {avg_adamw_time / PRINT_INTERVAL:.4f}s ({avg_adamw_time / avg_time * 100:.1f}%)"
            )

            avg_loss = 0.0
            avg_time = 0.0
            avg_data_time = 0.0
            avg_vjp_time = 0.0
            avg_adamw_time = 0.0

    print("\nNabla Modular training completed!")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    x_test_np = np.linspace(0, 1, 1000).reshape(-1, 1).astype(np.float32)
    targets_test_np = (
        np.sin(SIN_PERIODS * 2.0 * np.pi * x_test_np) / 2.0 + 0.5
    ).astype(np.float32)

    x_test = nb.Array.from_numpy(x_test_np)
    predictions_test = mlp_forward(x_test, params)

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
    final_loss, correlation = test_nabla_modular_mlp()
    print("\n=== Nabla Modular Summary ===")
    print(f"Final test loss: {final_loss:.6f}")
    print(f"Correlation with true function: {correlation:.4f}")

    if correlation > 0.95:
        print("SUCCESS: Nabla Modular learned the complex function very well! 🎉")
    elif correlation > 0.8:
        print("GOOD: Nabla Modular learned the general shape well! 👍")
    elif correlation > 0.5:
        print("PARTIAL: Some learning but needs improvement 🤔")
    else:
        print("POOR: Nabla Modular failed to learn the complex function 😞")
