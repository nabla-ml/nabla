#!/usr/bin/env python3

"""Test floor and floor division operations."""

import numpy as np

import nabla as nb


def test_floor():
    """Test the floor operation."""
    print("Testing floor operation...")

    # Test with positive numbers
    x = nb.array([1.2, 2.7, 3.0, 4.9])
    result = nb.floor(x)
    expected = np.array([1.0, 2.0, 3.0, 4.0])

    print(f"Input: {x.to_numpy()}")
    print(f"Floor result: {result.to_numpy()}")
    print(f"Expected: {expected}")
    print(f"Match: {np.allclose(result.to_numpy(), expected)}")
    print()

    # Test with negative numbers
    x_neg = nb.array([-1.2, -2.7, -3.0, -4.9])
    result_neg = nb.floor(x_neg)
    expected_neg = np.array([-2.0, -3.0, -3.0, -5.0])

    print(f"Input (negative): {x_neg.to_numpy()}")
    print(f"Floor result: {result_neg.to_numpy()}")
    print(f"Expected: {expected_neg}")
    print(f"Match: {np.allclose(result_neg.to_numpy(), expected_neg)}")
    print()


def test_floordiv():
    """Test the floor division operation."""
    print("Testing floor division operation...")

    # Test with positive numbers
    a = nb.array([7.0, 8.0, 9.0, 10.0])
    b = nb.array([3.0, 3.0, 3.0, 3.0])
    result = nb.floordiv(a, b)
    expected = np.array([2.0, 2.0, 3.0, 3.0])

    print(f"a: {a.to_numpy()}")
    print(f"b: {b.to_numpy()}")
    print(f"a // b: {result.to_numpy()}")
    print(f"Expected: {expected}")
    print(f"Match: {np.allclose(result.to_numpy(), expected)}")
    print()

    # Test with negative numbers (floor division rounds toward negative infinity)
    a_neg = nb.array([-7.0, -8.0, -9.0, -10.0])
    b_pos = nb.array([3.0, 3.0, 3.0, 3.0])
    result_neg = nb.floordiv(a_neg, b_pos)
    expected_neg = np.array([-3.0, -3.0, -3.0, -4.0])

    print(f"a (negative): {a_neg.to_numpy()}")
    print(f"b: {b_pos.to_numpy()}")
    print(f"a // b: {result_neg.to_numpy()}")
    print(f"Expected: {expected_neg}")
    print(f"Match: {np.allclose(result_neg.to_numpy(), expected_neg)}")
    print()


def test_floordiv_operator():
    """Test the // operator."""
    print("Testing // operator...")

    a = nb.array([7.5, 8.2, 9.1, 10.9])
    b = nb.array([3.0, 3.0, 3.0, 3.0])
    result = a // b
    expected = np.array([2.0, 2.0, 3.0, 3.0])

    print(f"a: {a.to_numpy()}")
    print(f"b: {b.to_numpy()}")
    print(f"a // b: {result.to_numpy()}")
    print(f"Expected: {expected}")
    print(f"Match: {np.allclose(result.to_numpy(), expected)}")
    print()


if __name__ == "__main__":
    test_floor()
    test_floordiv()
    test_floordiv_operator()
    print("All tests completed!")
