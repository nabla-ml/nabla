# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Mathematical Property Tests
# ===----------------------------------------------------------------------=== #

"""Tests for mathematical properties of operations."""

import numpy as np
import pytest
from tests_improved.utils.assertions import assert_arrays_close, get_tolerance_for_dtype
from tests_improved.utils.data_generators import (
    generate_test_array,
)

import nabla as nb


@pytest.mark.property
class TestMatmulProperties:
    """Test mathematical properties of matrix multiplication."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_associativity(self, dtype):
        """Test that (AB)C = A(BC) for compatible matrices."""
        # Generate compatible shapes
        a = generate_test_array((2, 3), dtype, seed=42)
        b = generate_test_array((3, 4), dtype, seed=43)
        c = generate_test_array((4, 5), dtype, seed=44)

        # Compute (AB)C
        ab = nb.matmul(a, b)
        abc_left = nb.matmul(ab, c)

        # Compute A(BC)
        bc = nb.matmul(b, c)
        abc_right = nb.matmul(a, bc)

        # Verify associativity
        rtol, atol = get_tolerance_for_dtype(dtype)
        # Use slightly looser tolerance due to accumulated floating point errors
        assert_arrays_close(
            abc_left,
            abc_right,
            rtol * 10,
            atol * 10,
            "Matrix multiplication should be associative",
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_identity_property(self, dtype):
        """Test that A * I = I * A = A for square matrices."""
        size = 3
        a = generate_test_array((size, size), dtype, seed=42)
        identity = nb.Array.from_numpy(np.eye(size, dtype=dtype))

        # Test A * I = A
        result_right = nb.matmul(a, identity)
        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(result_right, a, rtol, atol, "A * I should equal A")

        # Test I * A = A
        result_left = nb.matmul(identity, a)
        assert_arrays_close(result_left, a, rtol, atol, "I * A should equal A")

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_distributivity_over_addition(self, dtype):
        """Test that A(B + C) = AB + AC."""
        # Generate compatible shapes
        a = generate_test_array((2, 3), dtype, seed=42)
        b = generate_test_array((3, 4), dtype, seed=43)
        c = generate_test_array((3, 4), dtype, seed=44)

        # Compute A(B + C)
        b_plus_c = nb.add(b, c)
        left_side = nb.matmul(a, b_plus_c)

        # Compute AB + AC
        ab = nb.matmul(a, b)
        ac = nb.matmul(a, c)
        right_side = nb.add(ab, ac)

        # Verify distributivity
        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            left_side,
            right_side,
            rtol * 2,
            atol * 2,
            "Matrix multiplication should distribute over addition",
        )

    def test_matmul_zero_property(self):
        """Test that A * 0 = 0 * A = 0."""
        dtype = np.float32
        a = generate_test_array((2, 3), dtype, seed=42)
        zero_right = nb.Array.from_numpy(np.zeros((3, 4), dtype=dtype))
        zero_left = nb.Array.from_numpy(np.zeros((2, 2), dtype=dtype))

        # Test A * 0 = 0
        result_right = nb.matmul(a, zero_right)
        expected_right = np.zeros((2, 4), dtype=dtype)
        assert_arrays_close(result_right, expected_right)

        # Test 0 * A = 0
        result_left = nb.matmul(zero_left, a)
        expected_left = np.zeros((2, 3), dtype=dtype)
        assert_arrays_close(result_left, expected_left)


@pytest.mark.property
class TestArithmeticProperties:
    """Test mathematical properties of arithmetic operations."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4)])
    def test_addition_commutativity(self, dtype, shape):
        """Test that a + b = b + a."""
        a = generate_test_array(shape, dtype, seed=42)
        b = generate_test_array(shape, dtype, seed=43)

        result_ab = nb.add(a, b)
        result_ba = nb.add(b, a)

        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            result_ab, result_ba, rtol, atol, "Addition should be commutative"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    def test_addition_associativity(self, dtype, shape):
        """Test that (a + b) + c = a + (b + c)."""
        a = generate_test_array(shape, dtype, seed=42)
        b = generate_test_array(shape, dtype, seed=43)
        c = generate_test_array(shape, dtype, seed=44)

        # Compute (a + b) + c
        ab = nb.add(a, b)
        abc_left = nb.add(ab, c)

        # Compute a + (b + c)
        bc = nb.add(b, c)
        abc_right = nb.add(a, bc)

        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            abc_left, abc_right, rtol, atol, "Addition should be associative"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    def test_addition_identity(self, dtype, shape):
        """Test that a + 0 = a."""
        a = generate_test_array(shape, dtype, seed=42)
        zero = nb.Array.from_numpy(np.zeros(shape, dtype=dtype))

        result = nb.add(a, zero)

        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            result, a, rtol, atol, "Adding zero should not change the array"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    def test_multiplication_commutativity(self, dtype, shape):
        """Test that a * b = b * a for element-wise multiplication."""
        a = generate_test_array(shape, dtype, seed=42)
        b = generate_test_array(shape, dtype, seed=43)

        result_ab = nb.mul(a, b)
        result_ba = nb.mul(b, a)

        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            result_ab,
            result_ba,
            rtol,
            atol,
            "Element-wise multiplication should be commutative",
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    def test_multiplication_identity(self, dtype, shape):
        """Test that a * 1 = a."""
        a = generate_test_array(shape, dtype, seed=42)
        one = nb.Array.from_numpy(np.ones(shape, dtype=dtype))

        result = nb.mul(a, one)

        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            result, a, rtol, atol, "Multiplying by one should not change the array"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    def test_distributivity(self, dtype, shape):
        """Test that a * (b + c) = a * b + a * c."""
        a = generate_test_array(shape, dtype, seed=42)
        b = generate_test_array(shape, dtype, seed=43)
        c = generate_test_array(shape, dtype, seed=44)

        # Compute a * (b + c)
        b_plus_c = nb.add(b, c)
        left_side = nb.mul(a, b_plus_c)

        # Compute a * b + a * c
        ab = nb.mul(a, b)
        ac = nb.mul(a, c)
        right_side = nb.add(ab, ac)

        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            left_side,
            right_side,
            rtol * 2,
            atol * 2,
            "Multiplication should distribute over addition",
        )


@pytest.mark.property
class TestUnaryOperationProperties:
    """Test mathematical properties of unary operations."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sine_range(self, dtype):
        """Test that sin(x) is always in [-1, 1]."""
        # Generate test data over a wide range
        x = generate_test_array((100,), dtype, low=-10, high=10, seed=42)

        result = nb.sin(x)
        result_np = result.to_numpy()

        assert np.all(result_np >= -1.0), "sin(x) should be >= -1"
        assert np.all(result_np <= 1.0), "sin(x) should be <= 1"

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cosine_range(self, dtype):
        """Test that cos(x) is always in [-1, 1]."""
        x = generate_test_array((100,), dtype, low=-10, high=10, seed=42)

        result = nb.cos(x)
        result_np = result.to_numpy()

        assert np.all(result_np >= -1.0), "cos(x) should be >= -1"
        assert np.all(result_np <= 1.0), "cos(x) should be <= 1"

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exponential_positivity(self, dtype):
        """Test that exp(x) is always positive."""
        x = generate_test_array((50,), dtype, low=-10, high=10, seed=42)

        result = nb.exp(x)
        result_np = result.to_numpy()

        assert np.all(result_np > 0), "exp(x) should always be positive"

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_trigonometric_identity(self, dtype):
        """Test that sin²(x) + cos²(x) = 1."""
        x = generate_test_array((50,), dtype, low=-5, high=5, seed=42)

        sin_x = nb.sin(x)
        cos_x = nb.cos(x)

        sin_squared = nb.mul(sin_x, sin_x)
        cos_squared = nb.mul(cos_x, cos_x)
        identity_result = nb.add(sin_squared, cos_squared)

        expected = np.ones(x.shape, dtype=dtype)
        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            identity_result,
            expected,
            rtol * 10,
            atol * 10,
            "sin²(x) + cos²(x) should equal 1",
        )
