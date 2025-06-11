#!/usr/bin/env python3
"""Short test of MLP training with our simplified value_and_grad."""

import sys

sys.path.insert(0, "..")

# Monkey patch the configuration for a short test
import mlp_train
from mlp_train import test_nabla_complex_sin

mlp_train.NUM_EPOCHS = 5
mlp_train.LAYERS = [1, 8, 1]  # Smaller network
mlp_train.BATCH_SIZE = 16
mlp_train.PRINT_INTERVAL = 2

if __name__ == "__main__":
    print("=== SHORT TEST OF MLP TRAINING ===")
    try:
        final_loss, correlation = test_nabla_complex_sin()
        print(
            f"\nSUCCESS! Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}"
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
