#!/usr/bin/env python3
"""Diagnostic script to analyze MLP training issues."""

import numpy as np

# Import from the original file
from mlp_training import (
    BATCH_SIZE,
    LAYERS,
    SIN_PERIODS,
    create_sin_dataset,
    initialize_mlp_params,
    mlp_forward,
)

import nabla as nb


def analyze_gradient_flow(params, gradients):
    """Analyze gradient magnitudes to detect vanishing/exploding gradients."""
    print("\n=== Gradient Analysis ===")
    for i, (param, grad) in enumerate(zip(params, gradients, strict=False)):
        param_norm = np.linalg.norm(param.to_numpy())
        grad_norm = np.linalg.norm(grad.to_numpy())
        ratio = grad_norm / param_norm if param_norm > 0 else float('inf')

        layer_type = "weight" if i % 2 == 0 else "bias"
        layer_num = i // 2 + 1
        print(f"Layer {layer_num} {layer_type}: param_norm={param_norm:.6f}, grad_norm={grad_norm:.6f}, ratio={ratio:.6f}")

def analyze_activations(x, params):
    """Analyze activations to detect dead neurons."""
    print("\n=== Activation Analysis ===")
    output = x

    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        linear_output = nb.matmul(output, w) + b

        layer_num = i // 2 + 1
        linear_np = linear_output.to_numpy()
        print(f"Layer {layer_num} linear output: mean={linear_np.mean():.6f}, std={linear_np.std():.6f}, min={linear_np.min():.6f}, max={linear_np.max():.6f}")

        # Apply ReLU to all layers except the last
        if i < len(params) - 2:
            output = nb.relu(linear_output)
            relu_np = output.to_numpy()
            dead_ratio = (relu_np == 0).mean()
            print(f"Layer {layer_num} ReLU output: mean={relu_np.mean():.6f}, std={relu_np.std():.6f}, dead_ratio={dead_ratio:.3f}")
        else:
            output = linear_output

def analyze_target_function():
    """Analyze the complexity of the target function."""
    print("\n=== Target Function Analysis ===")
    x_test = np.linspace(0, 1, 1000)
    y_test = np.sin(SIN_PERIODS * 2.0 * np.pi * x_test) / 2.0 + 0.5

    print(f"Target function has {SIN_PERIODS} complete periods in [0,1]")
    print(f"Target range: [{y_test.min():.3f}, {y_test.max():.3f}]")
    print(f"Target mean: {y_test.mean():.3f}")
    print(f"Target std: {y_test.std():.3f}")

    # Check how much variation there is
    derivatives = np.diff(y_test)
    print(f"Max derivative magnitude: {np.abs(derivatives).max():.3f}")

def test_simpler_architecture():
    """Test if a simpler architecture works better."""
    print("\n=== Testing Simpler Architecture ===")

    # Try a much simpler network
    simple_layers = [1, 32, 32, 1]
    print(f"Simple architecture: {simple_layers}")

    # Initialize with He initialization for ReLU
    np.random.seed(42)
    params = []
    for i in range(len(simple_layers) - 1):
        fan_in = simple_layers[i]
        std = np.sqrt(2.0 / fan_in)  # He initialization

        w_np = np.random.normal(0.0, std, (fan_in, simple_layers[i + 1])).astype(np.float32)
        b_np = np.zeros((1, simple_layers[i + 1]), dtype=np.float32)

        w = nb.Array.from_numpy(w_np)
        b = nb.Array.from_numpy(b_np)
        params.extend([w, b])

    # Test forward pass
    x, targets = create_sin_dataset(BATCH_SIZE)
    predictions = mlp_forward(x, params)

    pred_np = predictions.to_numpy()
    target_np = targets.to_numpy()

    print(f"Predictions range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")
    print(f"Targets range: [{target_np.min():.3f}, {target_np.max():.3f}]")

    initial_loss = np.mean((pred_np - target_np)**2)
    print(f"Initial loss with simpler architecture: {initial_loss:.6f}")

def main():
    """Run comprehensive diagnosis."""
    print("=== MLP Training Diagnosis ===")

    # Analyze target function complexity
    analyze_target_function()

    # Test current architecture
    print(f"\n=== Current Architecture: {LAYERS} ===")
    params = initialize_mlp_params(LAYERS)
    x, targets = create_sin_dataset(BATCH_SIZE)

    # Analyze initial activations
    analyze_activations(x, params)

    # Get initial predictions and loss
    predictions = mlp_forward(x, params)
    pred_np = predictions.to_numpy()
    target_np = targets.to_numpy()
    initial_loss = np.mean((pred_np - target_np)**2)
    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Initial predictions range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")

    # Test simpler architecture
    test_simpler_architecture()

if __name__ == "__main__":
    main()
