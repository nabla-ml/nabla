# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Improved Binary Operations Tests
# ===----------------------------------------------------------------------=== #

"""Improved unit tests for binary operations using pytest best practices."""

import numpy as np
import pytest
from tests_improved.utils.assertions import assert_tensor_close
from tests_improved.utils.data_generators import generate_test_array

import nabla as nb


class TestBinaryOperations:
    """Test class for binary operations."""

    @pytest.mark.parametrize(
        "op_name,x_data,y_data",
        [
            (
                "add",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (2, 3), "dtype": "float32"},
            ),
            (
                "add",
                {"shape": (5,), "dtype": "float64"},
                {"shape": (5,), "dtype": "float64"},
            ),
            (
                "add",
                {"shape": (1, 1), "dtype": "float32"},
                {"shape": (1, 1), "dtype": "float32"},
            ),
            (
                "sub",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (2, 3), "dtype": "float32"},
            ),
            (
                "sub",
                {"shape": (5,), "dtype": "float64"},
                {"shape": (5,), "dtype": "float64"},
            ),
            (
                "sub",
                {"shape": (1, 1), "dtype": "float32"},
                {"shape": (1, 1), "dtype": "float32"},
            ),
            (
                "mul",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (2, 3), "dtype": "float32"},
            ),
            (
                "mul",
                {"shape": (5,), "dtype": "float64"},
                {"shape": (5,), "dtype": "float64"},
            ),
            (
                "mul",
                {"shape": (1, 1), "dtype": "float32"},
                {"shape": (1, 1), "dtype": "float32"},
            ),
            (
                "div",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (2, 3), "dtype": "float32", "for_binary_op_rhs": True},
            ),
            (
                "div",
                {"shape": (5,), "dtype": "float64"},
                {"shape": (5,), "dtype": "float64", "for_binary_op_rhs": True},
            ),
            (
                "div",
                {"shape": (1, 1), "dtype": "float32"},
                {"shape": (1, 1), "dtype": "float32", "for_binary_op_rhs": True},
            ),
        ],
        ids=lambda args: f"{args[0]}_{args[1]['shape']}_{args[1]['dtype']}",
    )
    def test_binary_operations_same_shape(self, op_name, x_data, y_data):
        """Test binary operations with same-shaped tensors."""
        x = generate_test_array(**x_data)
        y = generate_test_array(**y_data)

        if op_name == "add":
            result = nb.add(x, y)
            expected = x.data + y.data
        elif op_name == "sub":
            result = nb.sub(x, y)
            expected = x.data - y.data
        elif op_name == "mul":
            result = nb.mul(x, y)
            expected = x.data * y.data
        elif op_name == "div":
            result = nb.div(x, y)
            expected = x.data / y.data

        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "op_name,x_data,y_data",
        [
            (
                "add",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (1,), "dtype": "float32"},
            ),
            (
                "add",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (3,), "dtype": "float32"},
            ),
            (
                "add",
                {"shape": (2, 1, 3), "dtype": "float64"},
                {"shape": (1, 2, 1), "dtype": "float64"},
            ),
            (
                "sub",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (1,), "dtype": "float32"},
            ),
            (
                "sub",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (3,), "dtype": "float32"},
            ),
            (
                "sub",
                {"shape": (2, 1, 3), "dtype": "float64"},
                {"shape": (1, 2, 1), "dtype": "float64"},
            ),
            (
                "mul",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (1,), "dtype": "float32"},
            ),
            (
                "mul",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (3,), "dtype": "float32"},
            ),
            (
                "mul",
                {"shape": (2, 1, 3), "dtype": "float64"},
                {"shape": (1, 2, 1), "dtype": "float64"},
            ),
            (
                "div",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (1,), "dtype": "float32", "for_binary_op_rhs": True},
            ),
            (
                "div",
                {"shape": (2, 3), "dtype": "float32"},
                {"shape": (3,), "dtype": "float32", "for_binary_op_rhs": True},
            ),
            (
                "div",
                {"shape": (2, 1, 3), "dtype": "float64"},
                {"shape": (1, 2, 1), "dtype": "float64", "for_binary_op_rhs": True},
            ),
        ],
        ids=lambda args: f"{args[0]}_{args[1]['shape']}_vs_{args[2]['shape']}",
    )
    def test_binary_operations_broadcasting(self, op_name, x_data, y_data):
        """Test binary operations with broadcasting."""
        x = generate_test_array(**x_data)
        y = generate_test_array(**y_data)

        if op_name == "add":
            result = nb.add(x, y)
            expected = x.data + y.data
        elif op_name == "sub":
            result = nb.sub(x, y)
            expected = x.data - y.data
        elif op_name == "mul":
            result = nb.mul(x, y)
            expected = x.data * y.data
        elif op_name == "div":
            result = nb.div(x, y)
            expected = x.data / y.data

        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "op_name,x_data,y_data",
        [
            (
                "power",
                {"shape": (2, 3), "dtype": "float32", "ensure_positive": True},
                {"shape": (2, 3), "dtype": "float32", "for_binary_op_rhs": True},
            ),
            (
                "power",
                {"shape": (5,), "dtype": "float64", "ensure_positive": True},
                {"shape": (5,), "dtype": "float64", "for_binary_op_rhs": True},
            ),
            (
                "power",
                {"shape": (1, 1), "dtype": "float32", "ensure_positive": True},
                {"shape": (1, 1), "dtype": "float32", "for_binary_op_rhs": True},
            ),
        ],
        ids=lambda args: f"{args[0]}_{args[1]['shape']}_{args[1]['dtype']}",
    )
    def test_power_operation(self, op_name, x_data, y_data):
        """Test power operation with positive bases."""
        x = generate_test_array(**x_data)
        y = generate_test_array(**y_data)

        result = nb.power(x, y)
        expected = np.power(x.data, y.data)
        assert_tensor_close(result, expected, rtol=1e-6)


class TestBinaryOperationsEdgeCases:
    """Test edge cases for binary operations."""

    def test_add_with_zeros(self):
        """Test addition with zero tensors."""
        x = nb.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype="float32")
        y = nb.Tensor(np.zeros((2, 2)), dtype="float32")
        result = nb.add(x, y)
        assert_tensor_close(result, x.data)

    def test_sub_identity(self):
        """Test subtraction of tensor from itself."""
        x = generate_test_array(shape=(2, 2), dtype="float32")
        result = nb.sub(x, x)
        expected = np.zeros((2, 2))
        assert_tensor_close(result, expected)

    def test_mul_with_ones(self):
        """Test multiplication with ones."""
        x = generate_test_array(shape=(2, 2), dtype="float32")
        y = nb.Tensor(np.ones((2, 2)), dtype="float32")
        result = nb.mul(x, y)
        assert_tensor_close(result, x.data)

    def test_mul_with_zeros(self):
        """Test multiplication with zeros."""
        x = generate_test_array(shape=(2, 2), dtype="float32")
        y = nb.Tensor(np.zeros((2, 2)), dtype="float32")
        result = nb.mul(x, y)
        expected = np.zeros((2, 2))
        assert_tensor_close(result, expected)

    def test_div_by_ones(self):
        """Test division by ones."""
        x = generate_test_array(shape=(2, 2), dtype="float32")
        y = nb.Tensor(np.ones((2, 2)), dtype="float32")
        result = nb.div(x, y)
        assert_tensor_close(result, x.data)

    def test_power_special_cases(self):
        """Test power operation with special cases."""
        # x^0 = 1
        x = generate_test_array(shape=(2, 2), dtype="float32", ensure_positive=True)
        y = nb.Tensor(np.zeros((2, 2)), dtype="float32")
        result = nb.power(x, y)
        expected = np.ones((2, 2))
        assert_tensor_close(result, expected)

        # x^1 = x
        y = nb.Tensor(np.ones((2, 2)), dtype="float32")
        result = nb.power(x, y)
        assert_tensor_close(result, x.data)

    def test_scalar_broadcasting(self):
        """Test operations with scalar-like tensors."""
        x = generate_test_array(shape=(2, 3), dtype="float32")
        scalar = nb.Tensor(np.array([2.0]), dtype="float32")

        result_add = nb.add(x, scalar)
        expected_add = x.data + 2.0
        assert_tensor_close(result_add, expected_add)

        result_mul = nb.mul(x, scalar)
        expected_mul = x.data * 2.0
        assert_tensor_close(result_mul, expected_mul)


class TestBinaryOperationsGradients:
    """Test gradients for binary operations."""

    @pytest.mark.parametrize(
        "op_name,op_func",
        [
            ("add", nb.add),
            ("sub", nb.sub),
            ("mul", nb.mul),
            ("div", nb.div),
        ],
    )
    def test_gradient_shapes(self, op_name, op_func):
        """Test that gradients have correct shapes."""
        x = generate_test_array(shape=(2, 3), dtype="float32", requires_grad=True)
        y_kwargs = {"for_binary_op_rhs": True} if op_name == "div" else {}
        y = generate_test_array(
            shape=(2, 3), dtype="float32", requires_grad=True, **y_kwargs
        )

        result = op_func(x, y)
        loss = nb.reduce_sum(result)
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.shape == x.shape
        assert y.grad.shape == y.shape
        assert x.grad.dtype == x.dtype
        assert y.grad.dtype == y.dtype

    def test_power_gradients(self):
        """Test gradients for power operation."""
        x = generate_test_array(
            shape=(2, 3), dtype="float32", requires_grad=True, ensure_positive=True
        )
        y = generate_test_array(
            shape=(2, 3), dtype="float32", requires_grad=True, for_binary_op_rhs=True
        )

        result = nb.power(x, y)
        loss = nb.reduce_sum(result)
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.shape == x.shape
        assert y.grad.shape == y.shape


class TestBinaryOperationsChaining:
    """Test chaining of binary operations."""

    def test_operation_chaining(self):
        """Test chaining multiple binary operations."""
        x = generate_test_array(shape=(2, 2), dtype="float32")
        y = generate_test_array(shape=(2, 2), dtype="float32")
        z = generate_test_array(shape=(2, 2), dtype="float32")

        # (x + y) * z
        result = nb.mul(nb.add(x, y), z)
        expected = (x.data + y.data) * z.data
        assert_tensor_close(result, expected, rtol=1e-6)

    def test_mixed_operations(self):
        """Test mixing different binary operations."""
        x = generate_test_array(shape=(2, 2), dtype="float32")
        y = generate_test_array(shape=(2, 2), dtype="float32", for_binary_op_rhs=True)

        # x^2 + x/y
        power_y = nb.Tensor(np.array([[2.0, 2.0], [2.0, 2.0]]), dtype="float32")
        result = nb.add(nb.power(x, power_y), nb.div(x, y))
        expected = np.power(x.data, 2.0) + (x.data / y.data)
        assert_tensor_close(result, expected, rtol=1e-6)
