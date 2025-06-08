#!/usr/bin/env python3
"""Comprehensive tests for permute functionality."""

import nabla as nb
import numpy as np
from nabla.ops.view import permute, move_axis_to_front, move_axis_from_front


def test_permute_basic():
    """Test basic permute functionality."""
    print("Testing basic permute functionality...")

    # Create a simple 3D array
    x = nb.ones((2, 3, 4))
    print(f"Original shape: {x.shape}")

    # Test permutation (2, 0, 1) - move last axis to front
    y = permute(x, (2, 0, 1))
    print(f"After permute(x, (2, 0, 1)): {y.shape}")
    assert y.shape == (4, 2, 3), f"Expected (4, 2, 3), got {y.shape}"

    # Test identity permutation
    z = permute(x, (0, 1, 2))
    print(f"After identity permute: {z.shape}")
    assert z.shape == (2, 3, 4), f"Expected (2, 3, 4), got {z.shape}"

    # Test reverse permutation
    w = permute(x, (2, 1, 0))
    print(f"After reverse permute: {w.shape}")
    assert w.shape == (4, 3, 2), f"Expected (4, 3, 2), got {w.shape}"

    print("✓ Basic permute tests passed")


def test_move_axis_functions():
    """Test move_axis_to_front and move_axis_from_front."""
    print("Testing move axis functions...")

    # Create a 4D array
    x = nb.ones((2, 3, 4, 5))
    print(f"Original shape: {x.shape}")

    # Move axis 2 to front
    y = move_axis_to_front(x, 2)
    print(f"After move_axis_to_front(x, 2): {y.shape}")
    assert y.shape == (4, 2, 3, 5), f"Expected (4, 2, 3, 5), got {y.shape}"

    # Move front axis back to position 2
    z = move_axis_from_front(y, 2)
    print(f"After move_axis_from_front(y, 2): {z.shape}")
    assert z.shape == (2, 3, 4, 5), f"Expected (2, 3, 4, 5), got {z.shape}"

    # Test negative axis
    w = move_axis_to_front(x, -1)  # Move last axis to front
    print(f"After move_axis_to_front(x, -1): {w.shape}")
    assert w.shape == (5, 2, 3, 4), f"Expected (5, 2, 3, 4), got {w.shape}"

    print("✓ Move axis functions tests passed")


def test_permute_edge_cases():
    """Test edge cases for permute functionality."""
    print("Testing permute edge cases...")

    # Test 1D array
    x1d = nb.ones((5,))
    y1d = permute(x1d, (0,))
    print(f"1D array: {x1d.shape} -> {y1d.shape}")
    assert y1d.shape == (5,), f"Expected (5,), got {y1d.shape}"

    # Test 2D array
    x2d = nb.ones((3, 4))
    y2d = permute(x2d, (1, 0))  # Transpose
    print(f"2D array transpose: {x2d.shape} -> {y2d.shape}")
    assert y2d.shape == (4, 3), f"Expected (4, 3), got {y2d.shape}"

    # Test large permutation
    x5d = nb.ones((2, 3, 4, 5, 6))
    y5d = permute(x5d, (4, 2, 0, 3, 1))
    print(f"5D array permute: {x5d.shape} -> {y5d.shape}")
    assert y5d.shape == (6, 4, 2, 5, 3), f"Expected (6, 4, 2, 5, 3), got {y5d.shape}"

    print("✓ Edge cases passed")


def test_move_axis_edge_cases():
    """Test edge cases for move_axis functions."""
    print("Testing move_axis edge cases...")

    # Test moving first axis to front (should be no-op)
    x = nb.ones((2, 3, 4))
    y = move_axis_to_front(x, 0)
    print(f"Move axis 0 to front: {x.shape} -> {y.shape}")
    assert y.shape == x.shape, f"Should be unchanged"

    # Test moving from front to front (should be no-op)
    z = move_axis_from_front(y, 0)
    print(f"Move from front to position 0: {y.shape} -> {z.shape}")
    assert z.shape == y.shape, f"Should be unchanged"

    # Test negative indices
    w = move_axis_to_front(x, -2)  # Move second-to-last to front
    print(f"Move axis -2 to front: {x.shape} -> {w.shape}")
    assert w.shape == (3, 2, 4), f"Expected (3, 2, 4), got {w.shape}"

    print("✓ Move axis edge cases passed")


def test_permute_data_correctness():
    """Test that permute actually moves data to correct positions."""
    print("Testing permute data correctness...")

    # Create a 3D array with unique values at each position
    data = np.arange(24).reshape(2, 3, 4)
    x = nb.array(data)
    print(f"Original array shape: {x.shape}")

    # Test permutation (2, 0, 1) - move axis 2 to front
    y = permute(x, (2, 0, 1))
    print(f"After permute(2, 0, 1) shape: {y.shape}")

    # Verify data is in correct positions
    # Original: x[i, j, k] should become y[k, i, j]
    y_np = y.to_numpy()
    x_np = x.to_numpy()

    for i in range(2):
        for j in range(3):
            for k in range(4):
                original_val = x_np[i, j, k]
                new_val = y_np[k, i, j]
                assert original_val == new_val, (
                    f"Mismatch: x[{i},{j},{k}]={original_val} != y[{k},{i},{j}]={new_val}"
                )

    print("✓ Permute data correctness verified")


def test_move_axis_data_correctness():
    """Test that move_axis functions move data correctly."""
    print("Testing move_axis data correctness...")

    # Create a 3D array with unique values
    data = np.array(
        [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
    )  # Shape (2, 3, 2)
    x = nb.array(data)
    print(f"Original array shape: {x.shape}")

    # Move axis 2 to front: (2, 3, 2) -> (2, 2, 3)
    y = move_axis_to_front(x, 2)
    print(f"After move_axis_to_front(x, 2) shape: {y.shape}")

    # Verify: x[i, j, k] should become y[k, i, j]
    x_np = x.to_numpy()
    y_np = y.to_numpy()

    for i in range(2):
        for j in range(3):
            for k in range(2):
                original_val = x_np[i, j, k]
                new_val = y_np[k, i, j]
                assert original_val == new_val, (
                    f"Mismatch: x[{i},{j},{k}]={original_val} != y[{k},{i},{j}]={new_val}"
                )

    # Move it back: y[k, i, j] should become z[i, j, k]
    z = move_axis_from_front(y, 2)
    print(f"After move_axis_from_front(y, 2) shape: {z.shape}")

    # Should be identical to original
    z_np = z.to_numpy()
    np.testing.assert_array_equal(x_np, z_np)

    print("✓ Move axis data correctness verified")


def test_vmap_scenario_data():
    """Test the specific vmap scenario with actual data."""
    print("Testing vmap scenario with real data...")

    # Create arrays like in the vmap problem
    # z has batch dimension at axis 2
    z_data = np.arange(120).reshape(4, 5, 6)  # (C=4, D=5, K=6)
    z = nb.array(z_data)
    print(f"Original z shape: {z.shape}")

    # Move batch axis to front
    z_batched = move_axis_to_front(z, 2)
    print(f"After batching shape: {z_batched.shape}")

    # Verify the batch elements are correctly moved
    z_np = z.to_numpy()
    z_batched_np = z_batched.to_numpy()

    # z[c, d, k] should become z_batched[k, c, d]
    for c in range(4):
        for d in range(5):
            for k in range(6):
                original_val = z_np[c, d, k]
                batched_val = z_batched_np[k, c, d]
                assert original_val == batched_val, (
                    f"Batch move failed: z[{c},{d},{k}]={original_val} != z_batched[{k},{c},{d}]={batched_val}"
                )

    # Move back and verify round-trip
    z_restored = move_axis_from_front(z_batched, 2)
    z_restored_np = z_restored.to_numpy()
    np.testing.assert_array_equal(z_np, z_restored_np)

    print("✓ vmap scenario data correctness verified")


def test_permute_error_cases():
    """Test error conditions for permute."""
    print("Testing permute error cases...")

    x = nb.ones((2, 3, 4))

    # Test invalid permutation - wrong length
    try:
        permute(x, (0, 1))  # Too few axes
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error for wrong length: {e}")

    # Test invalid permutation - duplicate axes
    try:
        permute(x, (0, 1, 1))  # Duplicate axis
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error for duplicate axes: {e}")

    # Test invalid permutation - out of range
    try:
        permute(x, (0, 1, 3))  # Axis 3 doesn't exist
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error for out of range: {e}")

    print("✓ Error cases handled correctly")


def test_move_axis_error_cases():
    """Test error conditions for move_axis functions."""
    print("Testing move_axis error cases...")

    x = nb.ones((2, 3, 4))

    # Test out of range axis
    try:
        move_axis_to_front(x, 5)  # Axis 5 doesn't exist
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error for out of range axis: {e}")

    # Test out of range negative axis
    try:
        move_axis_to_front(x, -5)  # Too negative
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected error for negative out of range: {e}")

    print("✓ Move axis error cases handled correctly")


def run_all_tests():
    """Run all permute tests."""
    print("=" * 60)
    print("COMPREHENSIVE PERMUTE OPERATION TESTS")
    print("=" * 60)

    try:
        test_permute_basic()
        test_move_axis_functions()
        test_permute_edge_cases()
        test_move_axis_edge_cases()
        test_permute_data_correctness()
        test_move_axis_data_correctness()
        test_vmap_scenario_data()
        test_permute_error_cases()
        test_move_axis_error_cases()

        print("=" * 60)
        print("✅ ALL PERMUTE TESTS PASSED!")
        print("The permute operations are working correctly!")
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
