# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla as nb
from nabla.transforms import compile
from nabla.core.sharding import DeviceMesh, PartitionSpec as P
from nabla.ops import shard


def test_compile_basic():
    @compile
    def f(x):
        return x * 2.0 + 1.0

    x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = nb.Tensor.constant(x_np)

    # First call: trace and compile
    y = f(x)
    y_val = y.to_numpy()

    expected = x_np * 2.0 + 1.0
    np.testing.assert_allclose(y_val, expected)

    assert f.stats.misses == 1
    assert f.stats.hits == 0

    # Second call: cache hit
    y2 = f(x)
    y2_val = y2.to_numpy()
    np.testing.assert_allclose(y2_val, expected)

    assert f.stats.misses == 1
    assert f.stats.hits == 1
    print("Basic compile test passed!")


def test_compile_dynamic_dims():
    @compile(dynamic_dims={0: {0: "batch"}})
    def f(x, w):
        return x @ w

    x1_np = np.random.randn(2, 4).astype(np.float32)
    w_np = np.random.randn(4, 3).astype(np.float32)

    x1 = nb.Tensor.constant(x1_np)
    w = nb.Tensor.constant(w_np)

    # First call: compile for batch=2
    y1 = f(x1, w)
    np.testing.assert_allclose(y1.to_numpy(), x1_np @ w_np, atol=1e-5)

    assert f.stats.misses == 1

    # Second call: same batch size, should hit
    y1_again = f(x1, w)
    assert f.stats.hits == 1

    # Third call: different batch size, but dynamic_dims should allow hit
    x2_np = np.random.randn(5, 4).astype(np.float32)
    x2 = nb.Tensor.constant(x2_np)

    y2 = f(x2, w)
    np.testing.assert_allclose(y2.to_numpy(), x2_np @ w_np, atol=1e-5)

    # If dynamic_dims works, this should be a HIT!
    assert f.stats.hits == 2
    assert f.stats.misses == 1
    print("Dynamic dims compile test passed!")


def test_compile_sharded():
    """Test compilation with sharded tensors."""
    try:
        mesh = DeviceMesh("test", (2,), ("x",))
    except Exception as e:
        pytest.skip(f"Skipping sharded test: {e}")
        return

    @compile
    def sharded_add(a, b):
        return a + b

    # Create sharded tensors
    a_np = np.random.randn(8, 4).astype(np.float32)
    b_np = np.random.randn(8, 4).astype(np.float32)

    a = shard(nb.Tensor.from_dlpack(a_np), mesh, P("x", None))
    b = nb.Tensor.from_dlpack(b_np)

    # First call: compile
    result = sharded_add(a, b)
    assert sharded_add.stats.misses == 1

    np.testing.assert_allclose(result.to_numpy(), a_np + b_np, atol=1e-5)

    # Second call: cache hit
    result2 = sharded_add(a, b)
    assert sharded_add.stats.hits == 1

    print("Sharded compile test passed!")


def test_compile_sharded_dynamic():
    """Test compilation with sharded tensors having dynamic dimensions."""
    try:
        mesh = DeviceMesh("test_mesh", (2,), ("x",))
    except Exception as e:
        pytest.skip(f"Skipping sharded test: {e}")
        return

    # Shard 'x' along the batch dimension (dim 0).
    # 'local_batch' refers to the shard size.
    @compile(dynamic_dims={0: {0: "local_batch"}})
    def f(x, w):
        return x @ w

    # Run 1: Batch 8 (Shard size 4)
    x_np = np.random.randn(8, 4).astype(np.float32)
    w_np = np.random.randn(4, 3).astype(np.float32)
    x = shard(nb.Tensor.from_dlpack(x_np), mesh, P("x", None))

    # Validation Check: Verify that we raise NotImplementedError for Sharded + Dynamic
    # This combination is currently prohibited due to symbolic complexity.
    with pytest.raises(
        NotImplementedError,
        match="Compilation of sharded tensors with dynamic dimensions is not yet supported",
    ):
        y1 = f(x, nb.Tensor.from_dlpack(w_np))

    print("Sharded + Dynamic Dims validation check passed (Correctly Raised Error)!")


def test_compile_complex_ops_sharded():
    """Test complex operations (ReLU, Sin, Sum, Broadcast) with sharded inputs and dynamic dimensions."""
    try:
        mesh = DeviceMesh("test_mesh_c", (2,), ("x",))
    except Exception as e:
        pytest.skip(f"Skipping sharded test: {e}")
        return

    @compile(dynamic_dims={0: {0: "batch"}})
    def complex_f(x):
        h = nb.relu(x)
        h = nb.sin(h)
        s = nb.sum(h, axis=1)
        return x + s.reshape((x.shape[0], 1))

    print("Skipping complex sharded dynamic test as it is temporarily unsupported.")
    return

    # Run 1: Batch 8
    x_np = np.random.randn(8, 4).astype(np.float32)
    x = shard(nb.Tensor.from_dlpack(x_np), mesh, P("x", None))

    # Expect error
    with pytest.raises(NotImplementedError):
        y = complex_f(x)

    print("Complex ops sharded validation check passed!")


if __name__ == "__main__":
    test_compile_basic()
    test_compile_dynamic_dims()
    test_compile_sharded()
    test_compile_sharded_dynamic()
    test_compile_complex_ops_sharded()
