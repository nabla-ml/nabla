#!/usr/bin/env python3
"""Basic operations test suite."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    from nabla import graph_improved as nabla

    # Create simple arrays
    x = nabla.array([[1.0, 2.0], [3.0, 4.0]])
    y = nabla.array([[5.0, 6.0], [7.0, 8.0]])

    # Test addition
    result_add = nabla.add(x, y)
    result_add.realize()
    assert result_add.shape == (2, 2)

    # Test multiplication
    result_mul = nabla.mul(x, y)
    result_mul.realize()
    assert result_mul.shape == (2, 2)

    # Test subtraction - sub operation not implemented yet
    # result_sub = nabla.sub(x, y)
    # result_sub.realize()
    # assert result_sub.shape == (2, 2)


def test_reduction_operations():
    """Test reduction operations."""
    from nabla import graph_improved as nabla

    # Create a test array
    x = nabla.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Test sum without axis
    result_sum_all = nabla.sum(x)
    result_sum_all.realize()
    assert result_sum_all.shape == ()

    # Test sum with axis
    result_sum_axis0 = nabla.sum(x, axes=0)
    result_sum_axis0.realize()
    assert result_sum_axis0.shape == (3,)

    result_sum_axis1 = nabla.sum(x, axes=1)
    result_sum_axis1.realize()
    assert result_sum_axis1.shape == (2,)


def test_matrix_operations():
    """Test matrix operations."""
    from nabla import graph_improved as nabla

    # Create matrices
    a = nabla.array([[1.0, 2.0], [3.0, 4.0]])
    b = nabla.array([[5.0, 6.0], [7.0, 8.0]])

    # Test matrix multiplication
    result_matmul = nabla.matmul(a, b)
    result_matmul.realize()
    assert result_matmul.shape == (2, 2)

    # Test transpose
    result_transpose = nabla.transpose(a)
    result_transpose.realize()
    assert result_transpose.shape == (2, 2)


def test_view_operations():
    """Test view operations."""
    from nabla import graph_improved as nabla

    # Create array
    x = nabla.array([[1.0, 2.0, 3.0, 4.0]])

    # Test reshape
    result_reshape = nabla.reshape(x, (2, 2))
    result_reshape.realize()
    assert result_reshape.shape == (2, 2)


def test_complex_operations():
    """Test complex combinations of operations."""
    from nabla import graph_improved as nabla

    # Test the example that was mentioned as working
    n0 = nabla.randn((8, 8), seed=42)
    n1 = nabla.randn((4, 8, 8), seed=43)
    n = nabla.sum(
        nabla.reshape(nabla.sin(n0 + n1), shape=(2, 2, 8, 8)),
        axes=(0, 1, 2),
        keep_dims=False,
    )
    n.realize()
    assert n.shape == (8,)


if __name__ == "__main__":
    test_basic_arithmetic()
    test_reduction_operations()
    test_matrix_operations()
    test_view_operations()
    test_complex_operations()
    print("✅ All basic operations tests passed!")
