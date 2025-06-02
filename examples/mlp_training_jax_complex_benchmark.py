#!/usr/bin/env python3
"""JAX benchmark implementation matching mlp_training_nabla_complex.py configuration."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

# Configuration - matching mlp_training_nabla_complex.py
BATCH_SIZE = 128
LAYERS = [1, 64, 128, 128, 64, 1]  # Wider, not deeper
LEARNING_RATE = 0.001  # Lower LR for stability
NUM_EPOCHS = 5000
PRINT_INTERVAL = 200
SIN_PERIODS = 8  # Keep the complex target!


def mlp_forward_leaky(params, x):
    """MLP forward pass with leaky ReLU to prevent dead neurons."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = jnp.dot(output, w) + b
        # Apply leaky ReLU to all layers except the last
        if i < len(params) - 2:
            output = jax.nn.leaky_relu(output, negative_slope=0.01)
    return output


def mean_squared_error(params, x, targets):
    """Compute mean squared error loss."""
    predictions = mlp_forward_leaky(params, x)
    diff = predictions - targets
    loss = jnp.mean(diff ** 2)
    return loss


def create_sin_dataset(batch_size=128, key=None):
    """Create the COMPLEX 8-period sin dataset."""
    if key is None:
        key = jax.random.PRNGKey(42)
    
    x = jax.random.uniform(key, (batch_size, 1), dtype=jnp.float32)
    targets = (jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x) / 2.0 + 0.5).astype(jnp.float32)
    return x, targets


def initialize_for_complex_function(layers, seed=42):
    """Initialize specifically for learning complex high-frequency functions."""
    key = jax.random.PRNGKey(seed)
    params = []

    for i in range(len(layers) - 1):
        fan_in, fan_out = layers[i], layers[i + 1]
        
        # Split key for each layer
        key, subkey = jax.random.split(key)

        if i == 0:  # First layer - needs to capture high frequency
            # Larger weights for first layer to capture high frequency patterns
            std = jnp.sqrt(4.0 / fan_in)
        elif i == len(layers) - 2:  # Output layer
            # Conservative output layer
            std = jnp.sqrt(0.5 / fan_in)
        else:  # Hidden layers
            # Standard He initialization
            std = jnp.sqrt(2.0 / fan_in)

        w = jax.random.normal(subkey, (fan_in, fan_out)) * std

        # Bias initialization strategy
        if i < len(layers) - 2:  # Hidden layers
            # Small positive bias to help with leaky ReLU
            b = jnp.ones((1, fan_out)) * 0.05
        else:  # Output layer
            # Initialize output bias to middle of target range
            b = jnp.ones((1, fan_out)) * 0.5

        params.extend([w, b])

    return params


def learning_rate_schedule(epoch, initial_lr=0.001, decay_factor=0.95, decay_every=1000):
    """Learning rate schedule for complex function learning."""
    return initial_lr * (decay_factor ** (epoch // decay_every))


def analyze_jax_learning_progress(params, epoch):
    """Analyze how well we're learning the complex function."""
    # Create a dense test set
    x_test = jnp.linspace(0, 1, 1000).reshape(-1, 1).astype(jnp.float32)
    targets_test = (jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x_test) / 2.0 + 0.5).astype(jnp.float32)

    predictions_test = mlp_forward_leaky(params, x_test)
    test_loss = jnp.mean((predictions_test - targets_test) ** 2)

    pred_range = jnp.max(predictions_test) - jnp.min(predictions_test)
    target_range = jnp.max(targets_test) - jnp.min(targets_test)
    range_ratio = pred_range / target_range

    print(f"  Test loss: {test_loss:.6f}, Range ratio: {range_ratio:.3f}")
    return test_loss


def test_jax_complex_sin():
    """Test JAX implementation for complex sin learning - matching nabla_complex setup."""
    print("=== Learning COMPLEX 8-Period Sin Function with JAX (Complex Config) ===")
    print(f"Architecture: {LAYERS}")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print(f"Sin periods: {SIN_PERIODS}")
    print(f"Batch size: {BATCH_SIZE}")

    # Initialize for complex function learning
    params = initialize_for_complex_function(LAYERS)
    
    # Initialize AdamW optimizer - matching Nabla's AdamW setup exactly
    optimizer = optax.adamw(
        learning_rate=LEARNING_RATE,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.01
    )
    opt_state = optimizer.init(params)

    # JIT compile the training step for performance
    @jit
    def train_step(params, opt_state, x, targets, learning_rate):
        # Update optimizer with current learning rate
        current_optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.01
        )
        
        loss, grads = jax.value_and_grad(mean_squared_error)(params, x, targets)
        updates, new_opt_state = current_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Initial analysis
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    x_init, targets_init = create_sin_dataset(BATCH_SIZE, subkey)
    predictions_init = mlp_forward_leaky(params, x_init)
    initial_loss = jnp.mean((predictions_init - targets_init) ** 2)

    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Initial predictions range: [{jnp.min(predictions_init):.3f}, {jnp.max(predictions_init):.3f}]")
    print(f"Targets range: [{jnp.min(targets_init):.3f}, {jnp.max(targets_init):.3f}]")

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
        x, targets = create_sin_dataset(BATCH_SIZE, subkey)

        # Training step
        params, opt_state, loss = train_step(params, opt_state, x, targets, current_lr)

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
                best_test_loss = float(test_loss)
                print(f"  New best test loss: {best_test_loss:.6f}")

            avg_loss = 0.0
            avg_time = 0.0

    print("\nJAX training completed!")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    x_test = jnp.linspace(0, 1, 1000).reshape(-1, 1).astype(jnp.float32)
    targets_test = (jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x_test) / 2.0 + 0.5).astype(jnp.float32)

    predictions_test = mlp_forward_leaky(params, x_test)
    final_test_loss = float(jnp.mean((predictions_test - targets_test) ** 2))

    print(f"Final test loss: {final_test_loss:.6f}")
    print(f"Final predictions range: [{float(jnp.min(predictions_test)):.3f}, {float(jnp.max(predictions_test)):.3f}]")
    print(f"Target range: [{float(jnp.min(targets_test)):.3f}, {float(jnp.max(targets_test)):.3f}]")

    # Calculate correlation
    correlation = float(jnp.corrcoef(predictions_test.flatten(), targets_test.flatten())[0, 1])
    print(f"Prediction-target correlation: {correlation:.4f}")

    return final_test_loss, correlation


if __name__ == "__main__":
    final_loss, correlation = test_jax_complex_sin()
    print("\n=== JAX Summary (Complex Config) ===")
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
