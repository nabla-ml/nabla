# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Improved Unary Operations Tests
# ===----------------------------------------------------------------------=== #

"""Improved unit tests for unary operations using pytest best practices."""

import numpy as np
import pytest
from tests_improved.utils.assertions import assert_tensor_close
from tests_improved.utils.data_generators import generate_test_array

import nabla as nb


class TestUnaryOperations:
    """Test class for unary operations."""

    @pytest.mark.parametrize(
        "operation,input_data",
        [
            ("negate", {"shape": (2, 3), "dtype": "float32"}),
            ("negate", {"shape": (5,), "dtype": "float64"}),
            ("negate", {"shape": (2, 2, 2), "dtype": "float32"}),
            ("negate", {"shape": (1, 1), "dtype": "float64"}),
        ],
        ids=lambda op: f"{op[0]}_{op[1]['shape']}_{op[1]['dtype']}",
    )
    def test_negate_basic(self, operation, input_data):
        """Test negate operation with various shapes and dtypes."""
        x = generate_test_array(**input_data)
        result = nb.negate(x)
        expected = -x.data
        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "operation,input_data",
        [
            ("exp", {"shape": (2, 3), "dtype": "float32"}),
            ("exp", {"shape": (5,), "dtype": "float64"}),
            ("exp", {"shape": (2, 2, 2), "dtype": "float32"}),
            ("exp", {"shape": (1, 1), "dtype": "float64"}),
        ],
        ids=lambda op: f"{op[0]}_{op[1]['shape']}_{op[1]['dtype']}",
    )
    def test_exp_basic(self, operation, input_data):
        """Test exp operation with various shapes and dtypes."""
        x = generate_test_array(**input_data)
        result = nb.exp(x)
        expected = np.exp(x.data)
        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "operation,input_data",
        [
            ("log", {"shape": (2, 3), "dtype": "float32", "ensure_positive": True}),
            ("log", {"shape": (5,), "dtype": "float64", "ensure_positive": True}),
            ("log", {"shape": (2, 2, 2), "dtype": "float32", "ensure_positive": True}),
            ("log", {"shape": (1, 1), "dtype": "float64", "ensure_positive": True}),
        ],
        ids=lambda op: f"{op[0]}_{op[1]['shape']}_{op[1]['dtype']}",
    )
    def test_log_basic(self, operation, input_data):
        """Test log operation with positive inputs."""
        x = generate_test_array(**input_data)
        result = nb.log(x)
        expected = np.log(x.data)
        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "operation,input_data",
        [
            ("sin", {"shape": (2, 3), "dtype": "float32"}),
            ("sin", {"shape": (5,), "dtype": "float64"}),
            ("sin", {"shape": (2, 2, 2), "dtype": "float32"}),
            ("sin", {"shape": (1, 1), "dtype": "float64"}),
        ],
        ids=lambda op: f"{op[0]}_{op[1]['shape']}_{op[1]['dtype']}",
    )
    def test_sin_basic(self, operation, input_data):
        """Test sin operation with various shapes and dtypes."""
        x = generate_test_array(**input_data)
        result = nb.sin(x)
        expected = np.sin(x.data)
        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "operation,input_data",
        [
            ("cos", {"shape": (2, 3), "dtype": "float32"}),
            ("cos", {"shape": (5,), "dtype": "float64"}),
            ("cos", {"shape": (2, 2, 2), "dtype": "float32"}),
            ("cos", {"shape": (1, 1), "dtype": "float64"}),
        ],
        ids=lambda op: f"{op[0]}_{op[1]['shape']}_{op[1]['dtype']}",
    )
    def test_cos_basic(self, operation, input_data):
        """Test cos operation with various shapes and dtypes."""
        x = generate_test_array(**input_data)
        result = nb.cos(x)
        expected = np.cos(x.data)
        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "operation,input_data",
        [
            ("relu", {"shape": (2, 3), "dtype": "float32"}),
            ("relu", {"shape": (5,), "dtype": "float64"}),
            ("relu", {"shape": (2, 2, 2), "dtype": "float32"}),
            ("relu", {"shape": (1, 1), "dtype": "float64"}),
        ],
        ids=lambda op: f"{op[0]}_{op[1]['shape']}_{op[1]['dtype']}",
    )
    def test_relu_basic(self, operation, input_data):
        """Test ReLU operation with various shapes and dtypes."""
        x = generate_test_array(**input_data)
        result = nb.relu(x)
        expected = np.maximum(x.data, 0)
        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "operation,input_data",
        [
            ("transpose", {"shape": (2, 3), "dtype": "float32"}),
            ("transpose", {"shape": (5, 1), "dtype": "float64"}),
            ("transpose", {"shape": (2, 2, 2), "dtype": "float32"}),
            ("transpose", {"shape": (1, 1), "dtype": "float64"}),
        ],
        ids=lambda op: f"{op[0]}_{op[1]['shape']}_{op[1]['dtype']}",
    )
    def test_transpose_basic(self, operation, input_data):
        """Test transpose operation with various shapes and dtypes."""
        x = generate_test_array(**input_data)
        result = nb.transpose(x)
        expected = np.transpose(x.data)
        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "operation,input_data",
        [
            ("reduce_sum", {"shape": (2, 3), "dtype": "float32"}),
            ("reduce_sum", {"shape": (5,), "dtype": "float64"}),
            ("reduce_sum", {"shape": (2, 2, 2), "dtype": "float32"}),
            ("reduce_sum", {"shape": (1, 1), "dtype": "float64"}),
        ],
        ids=lambda op: f"{op[0]}_{op[1]['shape']}_{op[1]['dtype']}",
    )
    def test_reduce_sum_basic(self, operation, input_data):
        """Test reduce_sum operation with various shapes and dtypes."""
        x = generate_test_array(**input_data)
        result = nb.reduce_sum(x)
        expected = np.sum(x.data)
        assert_tensor_close(result, expected, rtol=1e-6)


class TestUnaryOperationsEdgeCases:
    """Test edge cases for unary operations."""

    def test_negate_zero(self):
        """Test negate with zero values."""
        x = nb.Tensor(np.zeros((2, 2)), dtype="float32")
        result = nb.negate(x)
        expected = np.zeros((2, 2))
        assert_tensor_close(result, expected)

    def test_exp_large_values(self):
        """Test exp with large values (but not causing overflow)."""
        x = nb.Tensor(np.array([1.0, 2.0, 3.0]), dtype="float32")
        result = nb.exp(x)
        expected = np.exp(np.array([1.0, 2.0, 3.0]))
        assert_tensor_close(result, expected, rtol=1e-6)

    def test_log_edge_values(self):
        """Test log with edge values."""
        x = nb.Tensor(np.array([1.0, np.e, 10.0]), dtype="float32")
        result = nb.log(x)
        expected = np.log(np.array([1.0, np.e, 10.0]))
        assert_tensor_close(result, expected, rtol=1e-6)

    def test_sin_cos_special_values(self):
        """Test sin and cos with special values."""
        x = nb.Tensor(np.array([0.0, np.pi / 2, np.pi]), dtype="float32")

        sin_result = nb.sin(x)
        sin_expected = np.sin(np.array([0.0, np.pi / 2, np.pi]))
        assert_tensor_close(sin_result, sin_expected, rtol=1e-6)

        cos_result = nb.cos(x)
        cos_expected = np.cos(np.array([0.0, np.pi / 2, np.pi]))
        assert_tensor_close(cos_result, cos_expected, rtol=1e-6)

    def test_relu_negative_values(self):
        """Test ReLU with negative values."""
        x = nb.Tensor(np.array([-1.0, -2.0, 0.0, 1.0, 2.0]), dtype="float32")
        result = nb.relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert_tensor_close(result, expected)

    def test_transpose_1d(self):
        """Test transpose with 1D tensor."""
        x = nb.Tensor(np.array([1, 2, 3]), dtype="float32")
        result = nb.transpose(x)
        expected = np.transpose(np.array([1, 2, 3]))
        assert_tensor_close(result, expected)

    def test_reduce_sum_empty_like(self):
        """Test reduce_sum with minimal tensor."""
        x = nb.Tensor(np.array([5.0]), dtype="float32")
        result = nb.reduce_sum(x)
        expected = 5.0
        assert_tensor_close(result, expected)


class TestUnaryOperationsGradients:
    """Test gradients for unary operations."""

    @pytest.mark.parametrize(
        "op_name,op_func",
        [
            ("negate", nb.negate),
            ("exp", nb.exp),
            ("sin", nb.sin),
            ("cos", nb.cos),
            ("relu", nb.relu),
        ],
    )
    def test_gradient_shapes(self, op_name, op_func):
        """Test that gradients have correct shapes."""
        x = generate_test_array(shape=(2, 3), dtype="float32", requires_grad=True)
        if op_name == "log":
            # Ensure positive values for log
            x = generate_test_array(
                shape=(2, 3), dtype="float32", requires_grad=True, ensure_positive=True
            )

        result = op_func(x)
        loss = nb.reduce_sum(result)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.dtype == x.dtype
