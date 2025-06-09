#!/usr/bin/env python3
"""Debug the VJP issue that's causing 65-dimensional arrays."""

import numpy as np

import nabla as nb

# Simple test case
BATCH_SIZE = 4
LAYERS = [1, 2, 1]  # Much simpler network


def simple_mlp_forward(x: nb.Array, params: list[nb.Array]) -> nb.Array:
    """Simple MLP forward pass."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nb.matmul(output, w) + b
        # Apply ReLU to all layers except the last
        if i < len(params) - 2:
            output = nb.relu(output)
    return output


def simple_loss(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Simple MSE loss."""
    diff = predictions - targets
    squared_errors = diff * diff
    loss = nb.sum(squared_errors) / BATCH_SIZE
    return loss


def simple_forward_and_loss(inputs: list[nb.Array]) -> list[nb.Array]:
    """Combined forward pass and loss computation."""
    x, targets, *params = inputs
    predictions = simple_mlp_forward(x, params)
    loss = simple_loss(predictions, targets)
    return [loss]


def test_simple_vjp():
    """Test VJP with simple case."""
    print("=== Testing Simple VJP ===")

    # Create simple data
    np_x = np.random.uniform(0.0, 1.0, (BATCH_SIZE, 1)).astype(np.float32)
    np_targets = np.random.uniform(0.0, 1.0, (BATCH_SIZE, 1)).astype(np.float32)

    x = nb.Array.from_numpy(np_x)
    targets = nb.Array.from_numpy(np_targets)

    # Simple parameter initialization
    w1_np = np.random.normal(0.0, 0.1, (1, 2)).astype(np.float32)
    b1_np = np.zeros((1, 2), dtype=np.float32)
    w2_np = np.random.normal(0.0, 0.1, (2, 1)).astype(np.float32)
    b2_np = np.zeros((1, 1), dtype=np.float32)

    w1 = nb.Array.from_numpy(w1_np)
    b1 = nb.Array.from_numpy(b1_np)
    w2 = nb.Array.from_numpy(w2_np)
    b2 = nb.Array.from_numpy(b2_np)

    params = [w1, b1, w2, b2]

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"W1 shape: {w1.shape}")
    print(f"B1 shape: {b1.shape}")
    print(f"W2 shape: {w2.shape}")
    print(f"B2 shape: {b2.shape}")

    # Test forward pass first
    predictions = simple_mlp_forward(x, params)
    print(f"Predictions shape: {predictions.shape}")

    loss = simple_loss(predictions, targets)
    print(f"Loss shape: {loss.shape}")
    print(f"Loss value: {loss.to_numpy().item()}")

    # Test VJP
    print("\n=== Testing VJP ===")
    all_inputs = [x, targets] + params
    print(f"Number of inputs to VJP: {len(all_inputs)}")

    try:
        loss_values, vjp_fn = nb.vjp(simple_forward_and_loss, all_inputs)
        print("VJP creation successful")
        print(f"Loss values: {[lv.shape for lv in loss_values]}")

        # Test different cotangent formats
        print("\n=== Testing different cotangent formats ===")

        # Format 1: List with array containing scalar
        cotangent1 = [nb.array([np.float32(1.0)])]
        print(
            f"Cotangent1 format: {type(cotangent1)} with {[c.shape for c in cotangent1]}"
        )

        try:
            all_gradients1 = vjp_fn(cotangent1)
            print(f"Cotangent1 SUCCESS: Got {len(all_gradients1)} gradients")
        except Exception as e:
            print(f"Cotangent1 FAILED: {e}")

        # Format 2: List with scalar array
        cotangent2 = [nb.array(np.float32(1.0))]
        print(
            f"Cotangent2 format: {type(cotangent2)} with {[c.shape for c in cotangent2]}"
        )

        try:
            all_gradients2 = vjp_fn(cotangent2)
            print(f"Cotangent2 SUCCESS: Got {len(all_gradients2)} gradients")
        except Exception as e:
            print(f"Cotangent2 FAILED: {e}")

        # Format 3: Just the scalar array
        cotangent3 = nb.array(np.float32(1.0))
        print(f"Cotangent3 format: {type(cotangent3)} with shape {cotangent3.shape}")

        try:
            all_gradients3 = vjp_fn([cotangent3])
            print(f"Cotangent3 SUCCESS: Got {len(all_gradients3)} gradients")
        except Exception as e:
            print(f"Cotangent3 FAILED: {e}")

    except Exception as e:
        print(f"VJP creation failed: {e}")


if __name__ == "__main__":
    test_simple_vjp()
