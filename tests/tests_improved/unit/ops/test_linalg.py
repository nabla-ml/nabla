# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Linear Algebra Operations Unit Tests (Improved)
# ===----------------------------------------------------------------------=== #

"""Improved unit tests for linear algebra operations."""

import numpy as np
import pytest
from tests_improved.utils.assertions import (
    assert_arrays_close,
    assert_gradients_close,
    assert_shapes_equal,
    get_tolerance_for_dtype,
)
from tests_improved.utils.data_generators import (
    MATMUL_COMPATIBLE_SHAPES,
    generate_test_array,
    generate_test_data_numpy,
)
from tests_improved.utils.fixtures import JAX_AVAILABLE, requires_jax

import nabla as nb

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp


class TestMatmulForwardPass:
    """Test matrix multiplication forward pass correctness."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            pytest.param(shapes[0], shapes[1], id=f"{shapes[0]}x{shapes[1]}")
            for shapes in MATMUL_COMPATIBLE_SHAPES
        ],
    )
    def test_matmul_against_numpy(self, dtype, shape_a, shape_b):
        """Test matrix multiplication against NumPy reference."""
        # Setup
        a_np = generate_test_data_numpy(shape_a, dtype, seed=42)
        b_np = generate_test_data_numpy(shape_b, dtype, seed=43)

        a_nb = nb.Array.from_numpy(a_np)
        b_nb = nb.Array.from_numpy(b_np)

        # Execute
        result_nb = nb.matmul(a_nb, b_nb)
        expected_np = np.matmul(a_np, b_np)

        # Verify
        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            result_nb,
            expected_np,
            rtol,
            atol,
            f"Matmul failed for shapes {shape_a} @ {shape_b}",
        )

    def test_matmul_output_shape(self):
        """Test that matmul produces correct output shapes."""
        test_cases = [
            ((2, 3), (3, 4), (2, 4)),
            ((1, 3), (3, 1), (1, 1)),
            ((5, 2, 3), (5, 3, 4), (5, 2, 4)),  # Batched
        ]

        for shape_a, shape_b, expected_shape in test_cases:
            a = generate_test_array(shape_a, np.float32)
            b = generate_test_array(shape_b, np.float32)

            result = nb.matmul(a, b)
            assert_shapes_equal(
                result, expected_shape, f"Wrong output shape for {shape_a} @ {shape_b}"
            )

    def test_matmul_dtype_preservation(self):
        """Test that matmul preserves input dtype."""
        for dtype in [np.float32, np.float64]:
            a = generate_test_array((2, 3), dtype)
            b = generate_test_array((3, 4), dtype)

            result = nb.matmul(a, b)
            result_np = result.to_numpy()

            assert result_np.dtype == dtype, (
                f"Expected dtype {dtype}, got {result_np.dtype}"
            )

    def test_matmul_edge_cases(self):
        """Test matmul with edge cases."""
        # Single element matrices
        a = nb.Array.from_numpy(np.array([[2.0]], dtype=np.float32))
        b = nb.Array.from_numpy(np.array([[3.0]], dtype=np.float32))
        result = nb.matmul(a, b)
        expected = np.array([[6.0]], dtype=np.float32)
        assert_arrays_close(result, expected)

        # Very small matrices
        a = nb.Array.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        b = nb.Array.from_numpy(np.array([[3.0], [4.0]], dtype=np.float32))
        result = nb.matmul(a, b)
        expected = np.array([[11.0]], dtype=np.float32)
        assert_arrays_close(result, expected)


class TestMatmulErrorHandling:
    """Test matrix multiplication error handling and validation."""

    def test_matmul_incompatible_shapes(self):
        """Test that incompatible shapes raise appropriate errors."""
        a = generate_test_array((2, 3), np.float32)
        b = generate_test_array((4, 5), np.float32)  # Incompatible inner dimensions

        with pytest.raises((ValueError, RuntimeError)):
            nb.matmul(a, b)

    def test_matmul_mismatched_batch_dimensions(self):
        """Test error handling for mismatched batch dimensions."""
        a = generate_test_array((2, 3, 4), np.float32)
        b = generate_test_array((5, 4, 6), np.float32)  # Different batch size

        with pytest.raises((ValueError, RuntimeError)):
            nb.matmul(a, b)

    def test_matmul_invalid_dimensions(self):
        """Test error handling for invalid tensor dimensions."""
        # 1D tensors should raise error or be handled specifically
        a = generate_test_array((5,), np.float32)
        b = generate_test_array((5,), np.float32)

        # Depending on implementation, this might raise an error or
        # be interpreted as dot product
        try:
            result = nb.matmul(a, b)
            # If it doesn't raise an error, check it behaves reasonably
            assert result.shape == () or result.shape == (1,)
        except (ValueError, RuntimeError):
            # This is also acceptable behavior
            pass


@requires_jax
class TestMatmulVsJAX:
    """Test matrix multiplication against JAX reference (when available)."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        "shape_a,shape_b", MATMUL_COMPATIBLE_SHAPES[:3]
    )  # Subset for speed
    def test_matmul_values_vs_jax(self, dtype, shape_a, shape_b):
        """Test matrix multiplication values against JAX."""
        # Setup
        a_np = generate_test_data_numpy(shape_a, dtype, seed=42)
        b_np = generate_test_data_numpy(shape_b, dtype, seed=43)

        # Nabla computation
        a_nb = nb.Array.from_numpy(a_np)
        b_nb = nb.Array.from_numpy(b_np)
        result_nb = nb.matmul(a_nb, b_nb)

        # JAX computation
        a_jax = jnp.array(a_np)
        b_jax = jnp.array(b_np)
        result_jax = jnp.matmul(a_jax, b_jax)

        # Compare
        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(result_nb, result_jax, rtol, atol)


@requires_jax
class TestMatmulGradients:
    """Test matrix multiplication gradients."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_vjp_vs_jax(self, dtype):
        """Test VJP for matrix multiplication against JAX."""
        # Use simple shapes for gradient testing
        shape_a, shape_b = (2, 3), (3, 4)

        # Setup
        a_np = generate_test_data_numpy(shape_a, dtype, seed=42)
        b_np = generate_test_data_numpy(shape_b, dtype, seed=43)

        primals_nb = [nb.Array.from_numpy(a_np), nb.Array.from_numpy(b_np)]
        primals_jax = [jnp.array(a_np), jnp.array(b_np)]

        # Create cotangent
        dummy_out = jnp.matmul(primals_jax[0], primals_jax[1])
        cotangent_np = generate_test_data_numpy(dummy_out.shape, dtype, seed=44)
        cotangent_nb = nb.Array.from_numpy(cotangent_np)
        cotangent_jax = jnp.array(cotangent_np)

        # Nabla VJP
        nabla_op = lambda inputs: [nb.matmul(inputs[0], inputs[1])]
        outputs_nb, vjp_fn_nb = nb.vjp(nabla_op, primals_nb)
        grads_nb = vjp_fn_nb([cotangent_nb])

        # JAX VJP
        jax_op = lambda x, y: jnp.matmul(x, y)
        outputs_jax, vjp_fn_jax = jax.vjp(jax_op, *primals_jax)
        grads_jax = vjp_fn_jax(cotangent_jax)

        # Compare gradients
        rtol, atol = get_tolerance_for_dtype(dtype, is_gradient=True)
        assert_gradients_close(grads_nb, grads_jax, rtol, atol)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_jvp_vs_jax(self, dtype):
        """Test JVP for matrix multiplication against JAX."""
        # Use simple shapes
        shape_a, shape_b = (2, 3), (3, 4)

        # Setup
        primals_np = [
            generate_test_data_numpy(shape_a, dtype, seed=42),
            generate_test_data_numpy(shape_b, dtype, seed=43),
        ]
        tangents_np = [
            generate_test_data_numpy(shape_a, dtype, seed=44),
            generate_test_data_numpy(shape_b, dtype, seed=45),
        ]

        primals_nb = [nb.Array.from_numpy(p) for p in primals_np]
        tangents_nb = [nb.Array.from_numpy(t) for t in tangents_np]
        primals_jax = [jnp.array(p) for p in primals_np]
        tangents_jax = [jnp.array(t) for t in tangents_np]

        # Nabla JVP
        nabla_op = lambda inputs: [nb.matmul(inputs[0], inputs[1])]
        primal_out_nb, tangent_out_nb = nb.jvp(nabla_op, primals_nb, tangents_nb)

        # JAX JVP
        jax_op = lambda x, y: jnp.matmul(x, y)
        primal_out_jax, tangent_out_jax = jax.jvp(
            jax_op, tuple(primals_jax), tuple(tangents_jax)
        )

        # Compare results
        rtol_val, atol_val = get_tolerance_for_dtype(dtype)
        rtol_tan, atol_tan = get_tolerance_for_dtype(dtype, is_gradient=True)

        assert_arrays_close(primal_out_nb[0], primal_out_jax, rtol_val, atol_val)
        assert_arrays_close(tangent_out_nb[0], tangent_out_jax, rtol_tan, atol_tan)


# Note: Performance tests would go in tests_improved/performance/
# Property tests would go in tests_improved/unit/properties/
