#!/usr/bin/env python3
"""Comprehensive rehydration tests for all operation types."""

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.graph.engine import GRAPH


def test_binary_with_broadcast():
    """Test binary op with broadcast (preprocessing in __call__)."""
    print("=" * 70)
    print("Test: Binary operation with broadcast (y = x1 + x2)")
    print("=" * 70)

    # Different shapes requiring broadcast
    x1 = nb.Tensor.from_dlpack(np.array([1.0, 2.0], dtype=np.float32))  # (2,)
    x2 = nb.Tensor.from_dlpack(
        np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    )  # (2, 2)

    GRAPH.evaluate(x1)
    GRAPH.evaluate(x2)

    def compute(a, b):
        result = nb.add(a, b)  # Should broadcast a to (2, 2)
        GRAPH.evaluate(result)
        return result

    traced = trace(compute, x1, x2)
    print(f"\nTrace:\n{traced}")

    # Check before rehydration
    print("\n--- Before Rehydration ---")
    for i, refs in enumerate(traced.nodes):
        alive = refs.get_alive_outputs()
        for j, impl in enumerate(alive):
            if impl:
                has_vals = bool(impl._get_valid_values())
                print(f"Node {i} ({refs.op.name}): has_values={has_vals}")

    # Rehydrate
    traced.rehydrate()

    # Check after rehydration
    print("\n--- After Rehydration ---")
    all_ok = True
    for i, refs in enumerate(traced.nodes):
        alive = refs.get_alive_outputs()
        for j, impl in enumerate(alive):
            if impl:
                has_vals = bool(impl._get_valid_values())
                print(f"Node {i} ({refs.op.name}): has_values={has_vals}")
                if not has_vals:
                    all_ok = False

    assert all_ok, "Some tensors not rehydrated"
    print("\n✓ SUCCESS: Binary with broadcast rehydrates correctly")


def test_matmul_1d_promotion():
    """Test matmul with 1D promotion (unsqueeze/squeeze in __call__)."""
    print("\n" + "=" * 70)
    print("Test: Matmul with 1D promotion")
    print("=" * 70)

    # 1D @ 2D case - should unsqueeze, matmul, then squeeze
    x1 = nb.Tensor.from_dlpack(np.array([1.0, 2.0], dtype=np.float32))  # (2,)
    x2 = nb.Tensor.from_dlpack(
        np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    )  # (2, 2)

    GRAPH.evaluate(x1)
    GRAPH.evaluate(x2)

    def compute(a, b):
        result = nb.matmul(a, b)  # Should unsqueeze, matmul, squeeze
        GRAPH.evaluate(result)
        return result

    traced = trace(compute, x1, x2)
    print(f"\nTrace:\n{traced}")

    # Rehydrate
    traced.rehydrate()

    # Check all nodes
    print("\n--- After Rehydration ---")
    all_ok = True
    for i, refs in enumerate(traced.nodes):
        alive = refs.get_alive_outputs()
        for j, impl in enumerate(alive):
            if impl:
                has_vals = bool(impl._get_valid_values())
                print(f"Node {i} ({refs.op.name}): has_values={has_vals}")
                if not has_vals:
                    all_ok = False

    assert all_ok, "Some tensors not rehydrated"
    print("\n✓ SUCCESS: Matmul with 1D promotion rehydrates correctly")


def test_reshape_operation():
    """Test reshape operation."""
    print("\n" + "=" * 70)
    print("Test: Reshape operation")
    print("=" * 70)

    x = nb.Tensor.from_dlpack(np.arange(12).astype(np.float32))  # (12,)
    GRAPH.evaluate(x)

    def compute(a):
        result = nb.ops.reshape(a, shape=(3, 4))
        GRAPH.evaluate(result)
        return result

    traced = trace(compute, x)
    print(f"\nTrace:\n{traced}")

    # Rehydrate
    traced.rehydrate()

    # Check
    print("\n--- After Rehydration ---")
    all_ok = True
    for i, refs in enumerate(traced.nodes):
        alive = refs.get_alive_outputs()
        for j, impl in enumerate(alive):
            if impl:
                has_vals = bool(impl._get_valid_values())
                print(f"Node {i} ({refs.op.name}): has_values={has_vals}")
                if not has_vals:
                    all_ok = False

    assert all_ok, "Some tensors not rehydrated"
    print("\n✓ SUCCESS: Reshape rehydrates correctly")


def test_reduction_operation():
    """Test reduction with axis kwargs."""
    print("\n" + "=" * 70)
    print("Test: Reduction operation")
    print("=" * 70)

    x = nb.Tensor.from_dlpack(np.random.randn(3, 4).astype(np.float32))
    GRAPH.evaluate(x)

    def compute(a):
        result = nb.ops.reduce_sum(a, axis=1, keepdims=False)
        GRAPH.evaluate(result)
        return result

    traced = trace(compute, x)
    print(f"\nTrace:\n{traced}")

    # Rehydrate
    traced.rehydrate()

    # Check
    print("\n--- After Rehydration ---")
    all_ok = True
    for i, refs in enumerate(traced.nodes):
        alive = refs.get_alive_outputs()
        for j, impl in enumerate(alive):
            if impl:
                has_vals = bool(impl._get_valid_values())
                print(f"Node {i} ({refs.op.name}): has_values={has_vals}")
                if not has_vals:
                    all_ok = False

    assert all_ok, "Some tensors not rehydrated"
    print("\n✓ SUCCESS: Reduction rehydrates correctly")


def test_composed_softmax():
    """Test composed operation (softmax decomposition)."""
    print("\n" + "=" * 70)
    print("Test: Composed softmax operation")
    print("=" * 70)

    x = nb.Tensor.from_dlpack(np.random.randn(3, 4).astype(np.float32))
    GRAPH.evaluate(x)

    def compute(a):
        # Softmax decomposes into multiple ops
        result = nb.ops.softmax(a, axis=-1)
        GRAPH.evaluate(result)
        return result

    traced = trace(compute, x)
    print(f"\nTrace:\n{traced}")
    print(f"Number of nodes: {len(traced.nodes)}")

    # Rehydrate
    traced.rehydrate()

    # Check
    print("\n--- After Rehydration ---")
    all_ok = True
    for i, refs in enumerate(traced.nodes):
        alive = refs.get_alive_outputs()
        for j, impl in enumerate(alive):
            if impl:
                has_vals = bool(impl._get_valid_values())
                op_name = refs.op.name if hasattr(refs.op, "name") else str(refs.op)
                print(f"Node {i} ({op_name}): has_values={has_vals}")
                if not has_vals:
                    all_ok = False

    assert all_ok, "Some tensors not rehydrated"
    print("\n✓ SUCCESS: Composed softmax rehydrates correctly")


if __name__ == "__main__":
    test_binary_with_broadcast()
    test_matmul_1d_promotion()
    test_reshape_operation()
    test_reduction_operation()
    test_composed_softmax()
    print("\n" + "=" * 70)
    print("ALL COMPREHENSIVE REHYDRATION TESTS PASSED!")
    print("=" * 70)
