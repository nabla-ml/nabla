# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Linear Algebra Operations Tests (Improved Structure)
# ===----------------------------------------------------------------------=== #

"""Improved unit tests for linear algebra operations."""

import numpy as np
import pytest

from tests.unit.test_utils import JAX_AVAILABLE, allclose_recursive, generate_test_data

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

import nabla as nb

# Test data configurations
MATMUL_TEST_CASES = [
    pytest.param((2, 3), (3, 4), id="2x3_3x4"),
    pytest.param((2, 3, 4), (2, 4, 5), id="batch_2x3x4_2x4x5"),
]

DTYPES = [
    pytest.param(np.float32, id="float32"),
    pytest.param(np.float64, id="float64"),
]


class TestMatmulOperation:
    """Test matrix multiplication operations."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape_a,shape_b", MATMUL_TEST_CASES)
    def test_matmul_forward_pass(self, dtype, shape_a, shape_b):
        """Test matrix multiplication forward pass correctness."""
        # Setup
        a_np = generate_test_data(shape_a, dtype)
        b_np = generate_test_data(shape_b, dtype)

        # Execute
        result_nb = nb.matmul(nb.Array.from_numpy(a_np), nb.Array.from_numpy(b_np))
        expected_np = np.matmul(a_np, b_np)

        # Verify
        rtol, atol = self._get_tolerances(dtype)
        assert allclose_recursive(result_nb, expected_np, rtol, atol)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape_a,shape_b", MATMUL_TEST_CASES)
    def test_matmul_against_jax(self, dtype, shape_a, shape_b):
        """Test matrix multiplication against JAX reference implementation."""
        # Setup
        a_np = generate_test_data(shape_a, dtype)
        b_np = generate_test_data(shape_b, dtype)

        # Execute both implementations
        result_nb = nb.matmul(nb.Array.from_numpy(a_np), nb.Array.from_numpy(b_np))
        result_jax = jnp.matmul(jnp.array(a_np), jnp.array(b_np))

        # Verify
        rtol, atol = self._get_tolerances(dtype)
        assert allclose_recursive(result_nb, result_jax, rtol, atol)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_matmul_vjp_correctness(self, dtype):
        """Test VJP for matrix multiplication."""
        # Use simple shapes for gradient tests
        shape_a, shape_b = (2, 3), (3, 4)

        # Setup
        a_np = generate_test_data(shape_a, dtype)
        b_np = generate_test_data(shape_b, dtype)

        primals_nb = [nb.Array.from_numpy(a_np), nb.Array.from_numpy(b_np)]
        primals_jax = [jnp.array(a_np), jnp.array(b_np)]

        # Create cotangent
        dummy_out = jnp.matmul(primals_jax[0], primals_jax[1])
        cotangent_np = generate_test_data(dummy_out.shape, dtype)
        cotangent_nb = nb.Array.from_numpy(cotangent_np)
        cotangent_jax = jnp.array(cotangent_np)

        # Execute VJP
        def nabla_op(inputs):
            return [nb.matmul(inputs[0], inputs[1])]

        _, vjp_fn_nb = nb.vjp(nabla_op, primals_nb)
        grads_nb = vjp_fn_nb([cotangent_nb])

        def jax_op(x, y):
            return jnp.matmul(x, y)

        _, vjp_fn_jax = jax.vjp(jax_op, *primals_jax)
        grads_jax = vjp_fn_jax(cotangent_jax)

        # Verify gradients
        rtol, atol = self._get_tolerances(dtype, is_gradient=True)
        assert len(grads_nb) == 2
        for i, (grad_nb, grad_jax) in enumerate(zip(grads_nb, grads_jax, strict=False)):
            assert allclose_recursive(grad_nb, grad_jax, rtol, atol), (
                f"Gradient mismatch for input {i}"
            )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_matmul_jvp_correctness(self, dtype):
        """Test JVP for matrix multiplication."""
        # Use simple shapes for gradient tests
        shape_a, shape_b = (2, 3), (3, 4)

        # Setup
        primals_np = [
            generate_test_data(shape_a, dtype),
            generate_test_data(shape_b, dtype),
        ]
        tangents_np = [
            generate_test_data(shape_a, dtype),
            generate_test_data(shape_b, dtype),
        ]

        primals_nb = [nb.Array.from_numpy(p) for p in primals_np]
        tangents_nb = [nb.Array.from_numpy(t) for t in tangents_np]
        primals_jax = [jnp.array(p) for p in primals_np]
        tangents_jax = [jnp.array(t) for t in tangents_np]

        # Execute JVP
        def nabla_op(inputs):
            return [nb.matmul(inputs[0], inputs[1])]

        primal_out_nb, tangent_out_nb = nb.jvp(nabla_op, primals_nb, tangents_nb)

        def jax_op(x, y):
            return jnp.matmul(x, y)

        primal_out_jax, tangent_out_jax = jax.jvp(
            jax_op, tuple(primals_jax), tuple(tangents_jax)
        )

        # Verify results
        rtol_val, atol_val = self._get_tolerances(dtype)
        rtol_tan, atol_tan = self._get_tolerances(dtype, is_gradient=True)

        assert allclose_recursive(primal_out_nb[0], primal_out_jax, rtol_val, atol_val)
        assert allclose_recursive(
            tangent_out_nb[0], tangent_out_jax, rtol_tan, atol_tan
        )

    def test_matmul_shape_validation(self):
        """Test that matmul properly validates input shapes."""
        a = nb.Array.from_numpy(np.random.randn(2, 3))
        b = nb.Array.from_numpy(np.random.randn(4, 5))  # Incompatible shape

        with pytest.raises((ValueError, RuntimeError)):
            nb.matmul(a, b)

    def test_matmul_edge_cases(self):
        """Test matmul with edge cases."""
        # Empty matrices
        a = nb.Array.from_numpy(np.random.randn(0, 3))
        b = nb.Array.from_numpy(np.random.randn(3, 0))
        result = nb.matmul(a, b)
        assert result.shape == (0, 0)

        # Single element
        a = nb.Array.from_numpy(np.array([[2.0]]))
        b = nb.Array.from_numpy(np.array([[3.0]]))
        result = nb.matmul(a, b)
        expected = np.array([[6.0]])
        assert allclose_recursive(result, expected, 1e-7, 1e-8)

    @staticmethod
    def _get_tolerances(dtype, is_gradient=False):
        """Get appropriate tolerances for the given dtype."""
        if np.dtype(dtype).itemsize >= 8:  # float64
            return (1e-6, 1e-7) if is_gradient else (1e-7, 1e-8)
        else:  # float32
            return (1e-4, 1e-5) if is_gradient else (1e-5, 1e-6)


class TestMatmulProperties:
    """Test mathematical properties of matrix multiplication."""

    def test_matmul_associativity(self):
        """Test that (AB)C = A(BC) for compatible matrices."""
        dtype = np.float64
        a = nb.Array.from_numpy(generate_test_data((2, 3), dtype))
        b = nb.Array.from_numpy(generate_test_data((3, 4), dtype))
        c = nb.Array.from_numpy(generate_test_data((4, 5), dtype))

        # (AB)C
        ab = nb.matmul(a, b)
        abc_left = nb.matmul(ab, c)

        # A(BC)
        bc = nb.matmul(b, c)
        abc_right = nb.matmul(a, bc)

        rtol, atol = 1e-7, 1e-8
        assert allclose_recursive(abc_left, abc_right, rtol, atol)

    def test_matmul_identity_property(self):
        """Test that A * I = I * A = A."""
        dtype = np.float64
        size = 3
        a = nb.Array.from_numpy(generate_test_data((size, size), dtype))
        identity = nb.Array.from_numpy(np.eye(size, dtype=dtype))

        # A * I = A
        result_right = nb.matmul(a, identity)
        # I * A = A
        result_left = nb.matmul(identity, a)

        rtol, atol = 1e-7, 1e-8
        assert allclose_recursive(result_right, a, rtol, atol)
        assert allclose_recursive(result_left, a, rtol, atol)


# Performance/benchmark tests could go in a separate module
@pytest.mark.benchmark
class TestMatmulPerformance:
    """Performance tests for matrix multiplication (marked for conditional execution)."""

    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_matmul_performance_scaling(self, size):
        """Test performance scaling of matrix multiplication."""
        a = nb.Array.from_numpy(np.random.randn(size, size).astype(np.float32))
        b = nb.Array.from_numpy(np.random.randn(size, size).astype(np.float32))

        # This would ideally measure execution time
        result = nb.matmul(a, b)
        assert result.shape == (size, size)
