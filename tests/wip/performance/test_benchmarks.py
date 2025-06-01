# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Performance and Benchmark Tests
# ===----------------------------------------------------------------------=== #

"""Performance benchmarks and scaling tests for Nabla operations."""

import time

import numpy as np
import pytest
from tests_improved.utils.data_generators import generate_test_array
from tests_improved.utils.fixtures import JAX_AVAILABLE, requires_jax

import nabla as nb

if JAX_AVAILABLE:
    import jax.numpy as jnp


@pytest.mark.benchmark
class TestMatmulPerformance:
    """Performance tests for matrix multiplication."""

    @pytest.mark.parametrize("size", [32, 64, 128, 256])
    def test_matmul_scaling(self, size):
        """Test performance scaling of square matrix multiplication."""
        # Generate test matrices
        a = generate_test_array((size, size), np.float32, seed=42)
        b = generate_test_array((size, size), np.float32, seed=43)

        # Warm up
        for _ in range(3):
            _ = nb.matmul(a, b)

        # Measure execution time
        start_time = time.perf_counter()
        result = nb.matmul(a, b)
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        # Basic assertions
        assert result.shape == (size, size)

        # Performance should be reasonable (very loose bounds)
        # This is more for tracking regressions than absolute performance
        assert execution_time < 10.0, (
            f"Matmul {size}x{size} took {execution_time:.3f}s, seems too slow"
        )

        print(f"Matmul {size}x{size}: {execution_time:.4f}s")

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_batched_matmul_scaling(self, batch_size):
        """Test performance scaling of batched matrix multiplication."""
        matrix_size = 64

        a = generate_test_array(
            (batch_size, matrix_size, matrix_size), np.float32, seed=42
        )
        b = generate_test_array(
            (batch_size, matrix_size, matrix_size), np.float32, seed=43
        )

        # Warm up
        for _ in range(3):
            _ = nb.matmul(a, b)

        # Measure
        start_time = time.perf_counter()
        result = nb.matmul(a, b)
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        assert result.shape == (batch_size, matrix_size, matrix_size)
        print(
            f"Batched matmul ({batch_size}, {matrix_size}, {matrix_size}): {execution_time:.4f}s"
        )

    @requires_jax
    @pytest.mark.parametrize("size", [64, 128])
    def test_matmul_vs_jax_performance(self, size):
        """Compare matrix multiplication performance against JAX."""
        # Generate test data
        a_np = np.random.randn(size, size).astype(np.float32)
        b_np = np.random.randn(size, size).astype(np.float32)

        # Nabla setup
        a_nb = nb.Array.from_numpy(a_np)
        b_nb = nb.Array.from_numpy(b_np)

        # JAX setup
        a_jax = jnp.array(a_np)
        b_jax = jnp.array(b_np)

        # Warm up both implementations
        for _ in range(3):
            _ = nb.matmul(a_nb, b_nb)
            _ = jnp.matmul(a_jax, b_jax).block_until_ready()

        # Measure Nabla
        start_time = time.perf_counter()
        result_nb = nb.matmul(a_nb, b_nb)
        nabla_time = time.perf_counter() - start_time

        # Measure JAX
        start_time = time.perf_counter()
        result_jax = jnp.matmul(a_jax, b_jax).block_until_ready()
        jax_time = time.perf_counter() - start_time

        print(
            f"Size {size}x{size}: Nabla {nabla_time:.4f}s, JAX {jax_time:.4f}s, "
            f"Ratio: {nabla_time / jax_time:.2f}x"
        )

        # Just ensure both produce valid results
        assert result_nb.shape == result_jax.shape


@pytest.mark.benchmark
class TestUnaryOpPerformance:
    """Performance tests for unary operations."""

    @pytest.mark.parametrize(
        "op_name,nb_op",
        [
            ("sin", nb.sin),
            ("cos", nb.cos),
            ("exp", nb.exp),
        ],
    )
    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_unary_op_scaling(self, op_name, nb_op, size):
        """Test performance scaling of unary operations."""
        x = generate_test_array((size,), np.float32, low=-2, high=2, seed=42)

        # Warm up
        for _ in range(3):
            _ = nb_op(x)

        # Measure
        start_time = time.perf_counter()
        result = nb_op(x)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        throughput = size / execution_time  # elements per second

        assert result.shape == (size,)
        print(
            f"{op_name} on {size} elements: {execution_time:.4f}s, "
            f"{throughput:.0f} elements/sec"
        )


@pytest.mark.benchmark
class TestBinaryOpPerformance:
    """Performance tests for binary operations."""

    @pytest.mark.parametrize(
        "op_name,nb_op",
        [
            ("add", nb.add),
            ("mul", nb.mul),
            ("sub", nb.sub),
        ],
    )
    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_binary_op_scaling(self, op_name, nb_op, size):
        """Test performance scaling of binary operations."""
        a = generate_test_array((size,), np.float32, seed=42)
        b = generate_test_array((size,), np.float32, seed=43)

        # Warm up
        for _ in range(3):
            _ = nb_op(a, b)

        # Measure
        start_time = time.perf_counter()
        result = nb_op(a, b)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        throughput = size / execution_time

        assert result.shape == (size,)
        print(
            f"{op_name} on {size} elements: {execution_time:.4f}s, "
            f"{throughput:.0f} elements/sec"
        )


@pytest.mark.benchmark
class TestGradientPerformance:
    """Performance tests for gradient computations."""

    @requires_jax
    @pytest.mark.parametrize("size", [32, 64])
    def test_vjp_performance(self, size):
        """Test VJP performance for a simple function."""
        x = generate_test_array((size, size), np.float32, seed=42)
        cotangent = generate_test_array((size, size), np.float32, seed=43)

        # Define a simple function: f(x) = sin(x) * cos(x)
        def test_fn(inputs):
            x = inputs[0]
            sin_x = nb.sin(x)
            cos_x = nb.cos(x)
            return [nb.mul(sin_x, cos_x)]

        # Warm up
        for _ in range(3):
            output, vjp_fn = nb.vjp(test_fn, [x])
            _ = vjp_fn([cotangent])

        # Measure
        start_time = time.perf_counter()
        output, vjp_fn = nb.vjp(test_fn, [x])
        grad = vjp_fn([cotangent])
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        assert len(grad) == 1
        assert grad[0].shape == x.shape
        print(f"VJP {size}x{size}: {execution_time:.4f}s")

    @requires_jax
    def test_gradient_memory_usage(self):
        """Test that gradient computation doesn't explode memory usage."""
        # This is a basic test - in practice you'd use memory profiling tools
        size = 100

        x = generate_test_array((size, size), np.float32, seed=42)

        def complex_fn(inputs):
            x = inputs[0]
            # Chain multiple operations
            y = nb.sin(x)
            z = nb.cos(y)
            w = nb.exp(z)
            return [nb.sum(w)]

        # This should complete without running out of memory
        output, vjp_fn = nb.vjp(complex_fn, [x])
        cotangent = nb.Array.from_numpy(np.array(1.0, dtype=np.float32))
        grad = vjp_fn([cotangent])

        assert len(grad) == 1
        assert grad[0].shape == x.shape


# Memory usage tests could be added here with tools like memory_profiler
@pytest.mark.benchmark
class TestMemoryUsage:
    """Memory usage tests."""

    def test_large_array_creation(self):
        """Test creating large arrays doesn't cause issues."""
        # Create a reasonably large array
        size = 1000
        large_array = generate_test_array((size, size), np.float32, seed=42)

        # Perform some operations
        result = nb.sin(large_array)
        assert result.shape == (size, size)

        # Clean up (though Python GC should handle this)
        del large_array, result

    def test_operation_chaining_memory(self):
        """Test that chaining operations doesn't leak memory excessively."""
        x = generate_test_array((100, 100), np.float32, seed=42)

        # Chain many operations
        result = x
        for _ in range(10):
            result = nb.sin(result)
            result = nb.add(result, x)

        assert result.shape == x.shape
