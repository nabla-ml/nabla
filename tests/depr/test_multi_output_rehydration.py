#!/usr/bin/env python3
"""Test for multi-output Trace rehydration."""

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace


def test_multi_output_rehydration():
    """Test that rehydration works for ops with multiple outputs (split)."""
    print("=" * 70)
    print("Test: Rehydration of multi-output trace (a, b = split(x))")
    print("=" * 70)

    # Create input
    x_data = np.arange(12).reshape(3, 4).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_data)

    # Define and trace computation
    def compute(x):
        a, b = nb.ops.split(x, num_splits=2, axis=1)
        # Force evaluation to clear _values
        from nabla.core.graph.engine import GRAPH

        GRAPH.evaluate(a)
        GRAPH.evaluate(b)
        return a, b

    traced = trace(compute, x)

    print(f"\nTrace:")
    print(traced)

    # Check status before
    print(f"\n--- Before Rehydration ---")
    node = traced.nodes[0]
    op_name = node.op.name
    alive = node.get_alive_outputs()
    print(f"Op: {op_name}, Num outputs: {len(alive)}")
    for i, impl in enumerate(alive):
        print(f"  Output {i} has_values: {bool(impl._get_valid_values())}")

    # Rehydrate
    print(f"\n--- Running Rehydration ---")
    traced.rehydrate()

    # Check status after
    print(f"\n--- After Rehydration ---")
    all_ok = True
    for i, impl in enumerate(alive):
        has_vals = bool(impl._get_valid_values())
        print(f"  Output {i} has_values: {has_vals}")
        if not has_vals:
            all_ok = False

    if all_ok:
        print(f"\n✓ SUCCESS: Multi-output rehydration works!")
    else:
        print(f"\n✗ FAILURE: Some outputs were not hydrated")

    assert all_ok, "Some outputs were not hydrated"


if __name__ == "__main__":
    success = test_multi_output_rehydration()
    exit(0 if success else 1)
