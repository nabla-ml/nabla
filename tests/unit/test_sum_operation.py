#!/usr/bin/env python3
"""Test to understand ops.sum behavior."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_ops_sum_behavior():
    """Test what ops.sum does by default."""
    from nabla import graph_improved as nabla
    from max.graph import Graph, TensorType, DeviceRef, ops
    from max.dtype import DType
    import numpy as np

    print("Testing ops.sum behavior...")

    # Create a simple test case to understand ops.sum
    try:
        x = nabla.randn((2, 3, 4), seed=42)
        x.realize()

        print(f"Original shape: {x.shape}")
        print(
            f"Original tensor shape: {x.tensor_value.shape if hasattr(x, 'tensor_value') and x.tensor_value else 'No tensor_value'}"
        )

        # Let's look at what the MAX ops.sum function does
        # by examining its output shape
        from max.driver import CPU

        input_types = [
            TensorType(
                dtype=DType.float32,
                shape=(2, 3, 4),
                device=DeviceRef.from_device(CPU()),
            )
        ]

        with Graph("test_sum", input_types=input_types) as graph:
            input_symbol = graph.inputs[0]
            print(f"Input symbol shape: {input_symbol.shape}")

            # Test sum over axis 1
            sum_result = ops.sum(input_symbol, axis=1)
            print(f"After ops.sum(axis=1) shape: {sum_result.shape}")

            graph.output(sum_result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_ops_sum_behavior()
