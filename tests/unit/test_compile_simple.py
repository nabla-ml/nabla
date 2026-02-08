import nabla as nb
from nabla.transforms.compile import compile
import numpy as np
import pytest


def test_compile_scalar_add():
    """Test compilation of a simple scalar addition."""

    @compile
    def add(x, y):
        return x + y

    x = nb.Tensor.constant(1.0)
    y = nb.Tensor.constant(2.0)

    # First call: Trace and compile
    res1 = add(x, y)
    assert np.allclose(res1.numpy(), 3.0)
    assert add.stats.misses == 1
    assert add.stats.hits == 0

    # Second call: Cache hit
    res2 = add(x, y)
    assert np.allclose(res2.numpy(), 3.0)
    assert add.stats.hits == 1


def test_compile_matmul():
    """Test compilation of a simple matrix multiplication."""

    @compile
    def matmul(x, w):
        return x @ w

    x_np = np.random.randn(2, 4).astype(np.float32)
    w_np = np.random.randn(4, 3).astype(np.float32)

    x = nb.Tensor.constant(x_np)
    w = nb.Tensor.constant(w_np)

    # First call
    res1 = matmul(x, w)
    expected = x_np @ w_np
    assert np.allclose(res1.numpy(), expected, atol=1e-5)
    assert matmul.stats.misses == 1

    # Second call
    res2 = matmul(x, w)
    assert np.allclose(res2.numpy(), expected, atol=1e-5)
    assert matmul.stats.hits == 1


def test_compile_dynamic_dims():
    """Test compilation with dynamic dimensions."""

    @compile(dynamic_dims={0: {0: "batch"}})
    def square(x):
        return x * x

    # Batch size 2
    x1 = nb.Tensor.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    res1 = square(x1)
    assert np.allclose(res1.numpy(), x1.numpy() ** 2)
    assert square.stats.misses == 1

    # Batch size 3 (should hit cache because of symbolic batch dim)
    x2 = nb.Tensor.constant(np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32))
    res2 = square(x2)
    assert np.allclose(res2.numpy(), x2.numpy() ** 2)
    # Check if it was a hit or miss.
    # NOTE: If the signature includes SymbolicDim, it should match even with different sizes.
    assert square.stats.hits == 1
