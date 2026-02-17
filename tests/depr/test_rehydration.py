#!/usr/bin/env python3
"""Test for Trace rehydration."""

import numpy as np

import nabla as nb
from nabla.core.graph.tracing import trace


def test_simple_rehydration():
    """Test that rehydration populates _values for all intermediate tensors."""
    print("=" * 70)
    print("Test: Rehydration of simple trace (y = x1 * x2 + x1)")
    print("=" * 70)

    # Create and realize inputs
    x1 = nb.Tensor.from_dlpack(np.array([2.0, 3.0], dtype=np.float32))
    x2 = nb.Tensor.from_dlpack(np.array([4.0, 5.0], dtype=np.float32))

    print("\nInputs:")
    print(f"x1 = {x1}")
    print(f"x2 = {x2}")

    # Ensure inputs are realized
    from nabla.core.graph.engine import GRAPH

    GRAPH.evaluate(x1)
    GRAPH.evaluate(x2)

    print(
        f"\nInputs realized: x1._impl.is_realized = {x1._impl.is_realized}, x2._impl.is_realized = {x2._impl.is_realized}"
    )

    # Define and trace computation
    def compute(x1, x2):
        prod = nb.mul(x1, x2)
        result = nb.add(prod, x1)
        # Evaluate intermediate to clear _values
        GRAPH.evaluate(prod)
        GRAPH.evaluate(result)
        return result

    traced = trace(compute, x1, x2)

    print("\nTrace:")
    print(traced)

    # Check that intermediate tensors might not have _values
    print("\n--- Before Rehydration ---")
    for i, output_refs in enumerate(traced.nodes):
        alive = output_refs.get_alive_outputs()
        op_name = getattr(output_refs.op, "name", str(output_refs.op))
        for j, impl in enumerate(alive):
            if impl is not None:
                has_values = bool(impl._get_valid_values())
                print(f"Node {i} (op={op_name}), output {j}: has_values={has_values}")

    # Rehydrate
    print("\n--- Running Rehydration ---")
    traced.rehydrate()

    # Check that all tensors now have _values
    print("\n--- After Rehydration ---")
    all_hydrated = True
    for i, output_refs in enumerate(traced.nodes):
        alive = output_refs.get_alive_outputs()
        op_name = getattr(output_refs.op, "name", str(output_refs.op))
        for j, impl in enumerate(alive):
            if impl is not None:
                has_values = bool(impl._get_valid_values())
                print(f"Node {i} (op={op_name}), output {j}: has_values={has_values}")
                if not has_values:
                    all_hydrated = False

    if all_hydrated:
        print("\n✓ SUCCESS: All intermediate tensors are rehydrated!")
    else:
        print("\n✗ FAILURE: Some tensors are still not rehydrated")

    assert all_hydrated, "Some tensors are still not rehydrated"


if __name__ == "__main__":
    success = test_simple_rehydration()
    exit(0 if success else 1)
