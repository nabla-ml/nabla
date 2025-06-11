#!/usr/bin/env python3
"""Example showcasing the new MLPBuilder and modular components."""

import numpy as np

import nabla as nb
from nabla import value_and_grad
from nabla.nn import MLPBuilder, create_sin_dataset

# Configuration
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 500
PRINT_INTERVAL = 100
SIN_PERIODS = 8


def train_with_builder():
    """Demonstrate training with the new MLPBuilder."""
    print("=== Training with MLPBuilder ===")

    # Create MLP configuration using the builder pattern
    mlp_config = (
        MLPBuilder()
        .with_layers([1, 64, 128, 256, 128, 64, 1])
        .with_activation("relu")
        .with_init_method("mlp_specialized")  # Use the specialized init from original
        .with_seed(42)
        .build()
    )

    print(f"Architecture: {mlp_config['layers']}")
    print(f"Activation: {mlp_config['activation']}")
    print(f"Init method: {mlp_config['init_method']}")

    # Extract components
    params = mlp_config["params"]
    forward_fn = mlp_config["forward"]
    forward_and_loss_fn = mlp_config["forward_and_loss"]

    # Initialize optimizer state
    from nabla.nn import adamw_step, init_adamw_state, learning_rate_schedule

    m_states, v_states = init_adamw_state(params)

    # Initial evaluation
    x_init, targets_init = create_sin_dataset(BATCH_SIZE, SIN_PERIODS)
    predictions_init = forward_fn(x_init, params)

    from nabla.nn import mean_squared_error

    initial_loss = mean_squared_error(predictions_init, targets_init)
    print(f"Initial loss: {initial_loss.to_numpy().item():.6f}")

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = learning_rate_schedule(epoch, LEARNING_RATE)

        # Create batch
        x, targets = create_sin_dataset(BATCH_SIZE, SIN_PERIODS)
        all_inputs = [x, targets] + params

        # Compute loss and gradients
        loss_values, param_gradients = value_and_grad(forward_and_loss_fn, all_inputs)

        # Update parameters
        params, m_states, v_states = adamw_step(
            params, param_gradients, m_states, v_states, epoch, current_lr
        )

        if epoch % PRINT_INTERVAL == 0:
            loss_value = loss_values[0].to_numpy().item()
            print(f"Epoch {epoch:3d} | Loss: {loss_value:.6f}")

    # Final evaluation
    x_test_np = np.linspace(0, 1, 1000).reshape(-1, 1).astype(np.float32)
    targets_test_np = (
        np.sin(SIN_PERIODS * 2.0 * np.pi * x_test_np) / 2.0 + 0.5
    ).astype(np.float32)

    x_test = nb.Array.from_numpy(x_test_np)
    predictions_test = forward_fn(x_test, params)
    pred_final_np = predictions_test.to_numpy()

    final_test_loss = np.mean((pred_final_np - targets_test_np) ** 2)
    correlation = np.corrcoef(pred_final_np.flatten(), targets_test_np.flatten())[0, 1]

    print(f"Final test loss: {final_test_loss:.6f}")
    print(f"Correlation: {correlation:.4f}")

    return final_test_loss, correlation


def demonstrate_different_configs():
    """Show different MLP configurations."""
    print("\n=== Different MLP Configurations ===")

    # Configuration 1: Small MLP with tanh activation
    config1 = (
        MLPBuilder()
        .with_layers([1, 32, 32, 1])
        .with_activation("tanh")
        .with_final_activation("sigmoid")
        .with_init_method("xavier_normal")
        .build()
    )

    print(f"Config 1: {config1['layers']} with {config1['activation']}")

    # Configuration 2: Large MLP with He initialization
    config2 = (
        MLPBuilder()
        .with_layers([1, 128, 256, 512, 256, 128, 1])
        .with_activation("relu")
        .with_init_method("he_normal")
        .build()
    )

    print(f"Config 2: {config2['layers']} with {config2['activation']}")

    # Test forward passes
    x_test = nb.rand((32, 1), dtype=nb.DType.float32)

    pred1 = config1["forward"](x_test, config1["params"])
    pred2 = config2["forward"](x_test, config2["params"])

    print(
        f"Config 1 output range: [{pred1.to_numpy().min():.3f}, {pred1.to_numpy().max():.3f}]"
    )
    print(
        f"Config 2 output range: [{pred2.to_numpy().min():.3f}, {pred2.to_numpy().max():.3f}]"
    )


def showcase_modular_components():
    """Showcase individual modular components."""
    print("\n=== Modular Components Showcase ===")

    # Loss functions
    from nabla.nn.losses import huber_loss, mean_absolute_error, mean_squared_error

    pred = nb.array([[1.0, 2.0, 3.0]])
    target = nb.array([[1.5, 2.2, 2.8]])

    mse = mean_squared_error(pred, target)
    mae = mean_absolute_error(pred, target)
    huber = huber_loss(pred, target)

    print(f"MSE Loss: {mse.to_numpy().item():.4f}")
    print(f"MAE Loss: {mae.to_numpy().item():.4f}")
    print(f"Huber Loss: {huber.to_numpy().item():.4f}")

    # Different initializations
    from nabla.nn.init import he_normal, lecun_normal, xavier_normal

    shape = (64, 32)
    he_weights = he_normal(shape, seed=42)
    xavier_weights = xavier_normal(shape, seed=42)
    lecun_weights = lecun_normal(shape, seed=42)

    print(f"He normal std: {he_weights.to_numpy().std():.4f}")
    print(f"Xavier normal std: {xavier_weights.to_numpy().std():.4f}")
    print(f"LeCun normal std: {lecun_weights.to_numpy().std():.4f}")

    # Learning rate schedules
    from nabla.nn.optim.schedules import (
        cosine_annealing_schedule,
        exponential_decay_schedule,
        warmup_cosine_schedule,
    )

    exp_schedule = exponential_decay_schedule(0.001, 0.9, 100)
    cos_schedule = cosine_annealing_schedule(0.001, 1e-6, 1000)
    warmup_schedule = warmup_cosine_schedule(0.001, 100, 1000)

    epochs = [0, 100, 500, 1000]
    print("\nLearning Rate Schedules:")
    for epoch in epochs:
        print(
            f"Epoch {epoch:4d}: Exp={exp_schedule(epoch):.6f}, "
            f"Cos={cos_schedule(epoch):.6f}, Warmup={warmup_schedule(epoch):.6f}"
        )


if __name__ == "__main__":
    # Main training demonstration
    final_loss, correlation = train_with_builder()

    # Show different configurations
    demonstrate_different_configs()

    # Showcase modular components
    showcase_modular_components()

    print("\n=== Summary ===")
    print("✅ Modular MLP training completed successfully!")
    print("✅ All components are properly organized in nabla.nn")
    print("✅ Builder pattern provides convenient configuration")
    print("✅ Individual components can be used independently")
    print(f"Final performance: Loss={final_loss:.6f}, Correlation={correlation:.4f}")
