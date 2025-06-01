# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Improved Transformation Tests (vmap)
# ===----------------------------------------------------------------------=== #

"""Improved unit tests for vmap transformations using pytest best practices."""

import numpy as np
import pytest
from tests_improved.utils.assertions import assert_tensor_close
from tests_improved.utils.data_generators import generate_test_array

import nabla as nb


class TestVmapTransformations:
    """Test class for vmap transformations."""

    @pytest.mark.parametrize(
        "op_name,batch_size,input_shape",
        [
            ("add", 3, (2, 3)),
            ("add", 5, (4,)),
            ("sub", 3, (2, 3)),
            ("sub", 5, (4,)),
            ("mul", 3, (2, 3)),
            ("mul", 5, (4,)),
            ("exp", 3, (2, 3)),
            ("exp", 5, (4,)),
            ("sin", 3, (2, 3)),
            ("sin", 5, (4,)),
            ("cos", 3, (2, 3)),
            ("cos", 5, (4,)),
            ("relu", 3, (2, 3)),
            ("relu", 5, (4,)),
        ],
        ids=lambda args: f"vmap_{args[0]}_batch{args[1]}_shape{args[2]}",
    )
    def test_vmap_basic_operations(self, op_name, batch_size, input_shape):
        """Test vmap with various operations and batch sizes."""
        if op_name in ["add", "sub", "mul"]:
            # Binary operations
            def test_fn(x, y):
                if op_name == "add":
                    return nb.add(x, y)
                elif op_name == "sub":
                    return nb.sub(x, y)
                elif op_name == "mul":
                    return nb.mul(x, y)

            x_batch = generate_test_array(
                shape=(batch_size,) + input_shape, dtype="float32"
            )
            y_kwargs = {"for_binary_op_rhs": True} if op_name in ["div", "mul"] else {}
            y_batch = generate_test_array(
                shape=(batch_size,) + input_shape, dtype="float32", **y_kwargs
            )

            vmapped_fn = nb.vmap(test_fn)
            result = vmapped_fn(x_batch, y_batch)

            # Compute expected result manually
            expected_list = []
            for i in range(batch_size):
                x_i = nb.Tensor(x_batch.data[i], dtype="float32")
                y_i = nb.Tensor(y_batch.data[i], dtype="float32")
                if op_name == "add":
                    expected_i = nb.add(x_i, y_i)
                elif op_name == "sub":
                    expected_i = nb.sub(x_i, y_i)
                elif op_name == "mul":
                    expected_i = nb.mul(x_i, y_i)
                expected_list.append(expected_i.data)

            expected = np.stack(expected_list, axis=0)
            assert_tensor_close(result, expected, rtol=1e-6)

        else:
            # Unary operations
            def test_fn(x):
                if op_name == "exp":
                    return nb.exp(x)
                elif op_name == "sin":
                    return nb.sin(x)
                elif op_name == "cos":
                    return nb.cos(x)
                elif op_name == "relu":
                    return nb.relu(x)

            x_batch = generate_test_array(
                shape=(batch_size,) + input_shape, dtype="float32"
            )

            vmapped_fn = nb.vmap(test_fn)
            result = vmapped_fn(x_batch)

            # Compute expected result manually
            expected_list = []
            for i in range(batch_size):
                x_i = nb.Tensor(x_batch.data[i], dtype="float32")
                if op_name == "exp":
                    expected_i = nb.exp(x_i)
                elif op_name == "sin":
                    expected_i = nb.sin(x_i)
                elif op_name == "cos":
                    expected_i = nb.cos(x_i)
                elif op_name == "relu":
                    expected_i = nb.relu(x_i)
                expected_list.append(expected_i.data)

            expected = np.stack(expected_list, axis=0)
            assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "batch_size,input_shape",
        [
            (2, (3, 4)),
            (3, (2,)),
            (4, (1, 1)),
            (5, (2, 2, 2)),
        ],
        ids=lambda args: f"batch{args[0]}_shape{args[1]}",
    )
    def test_vmap_nested_operations(self, batch_size, input_shape):
        """Test vmap with nested operations."""

        def complex_fn(x, y):
            # (x + y) * sin(x)
            sum_result = nb.add(x, y)
            sin_result = nb.sin(x)
            return nb.mul(sum_result, sin_result)

        x_batch = generate_test_array(
            shape=(batch_size,) + input_shape, dtype="float32"
        )
        y_batch = generate_test_array(
            shape=(batch_size,) + input_shape, dtype="float32"
        )

        vmapped_fn = nb.vmap(complex_fn)
        result = vmapped_fn(x_batch, y_batch)

        # Compute expected result manually
        expected_list = []
        for i in range(batch_size):
            x_i = nb.Tensor(x_batch.data[i], dtype="float32")
            y_i = nb.Tensor(y_batch.data[i], dtype="float32")
            expected_i = complex_fn(x_i, y_i)
            expected_list.append(expected_i.data)

        expected = np.stack(expected_list, axis=0)
        assert_tensor_close(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_vmap_different_axes(self, axis):
        """Test vmap with different batch axes."""

        def simple_fn(x):
            return nb.exp(x)

        if axis == 0:
            x_batch = generate_test_array(shape=(3, 2, 4), dtype="float32")
        elif axis == 1:
            x_batch = generate_test_array(shape=(2, 3, 4), dtype="float32")
        else:  # axis == -1
            x_batch = generate_test_array(shape=(2, 4, 3), dtype="float32")

        vmapped_fn = nb.vmap(simple_fn, in_axes=axis, out_axes=axis)
        result = vmapped_fn(x_batch)

        # The result should have the batch dimension preserved at the specified axis
        assert result.shape[axis] == 3

        # Check a few individual results
        for i in range(3):
            if axis == 0:
                x_i = nb.Tensor(x_batch.data[i], dtype="float32")
                expected_i = nb.exp(x_i)
                assert_tensor_close(result.data[i], expected_i.data, rtol=1e-6)
            elif axis == 1:
                x_i = nb.Tensor(x_batch.data[:, i, :], dtype="float32")
                expected_i = nb.exp(x_i)
                assert_tensor_close(result.data[:, i, :], expected_i.data, rtol=1e-6)
            else:  # axis == -1
                x_i = nb.Tensor(x_batch.data[:, :, i], dtype="float32")
                expected_i = nb.exp(x_i)
                assert_tensor_close(result.data[:, :, i], expected_i.data, rtol=1e-6)


class TestVmapEdgeCases:
    """Test edge cases for vmap transformations."""

    def test_vmap_single_batch(self):
        """Test vmap with batch size of 1."""

        def simple_fn(x):
            return nb.sin(x)

        x_batch = generate_test_array(shape=(1, 2, 3), dtype="float32")
        vmapped_fn = nb.vmap(simple_fn)
        result = vmapped_fn(x_batch)

        # Should work the same as regular operation
        x_single = nb.Tensor(x_batch.data[0], dtype="float32")
        expected = nb.sin(x_single)

        assert result.shape == (1,) + expected.shape
        assert_tensor_close(result.data[0], expected.data, rtol=1e-6)

    def test_vmap_scalar_like(self):
        """Test vmap with scalar-like tensors."""

        def simple_fn(x):
            return nb.exp(x)

        x_batch = generate_test_array(shape=(3, 1), dtype="float32")
        vmapped_fn = nb.vmap(simple_fn)
        result = vmapped_fn(x_batch)

        assert result.shape == (3, 1)

        for i in range(3):
            x_i = nb.Tensor(x_batch.data[i], dtype="float32")
            expected_i = nb.exp(x_i)
            assert_tensor_close(result.data[i], expected_i.data, rtol=1e-6)

    def test_vmap_identity_function(self):
        """Test vmap with identity function."""

        def identity_fn(x):
            return x

        x_batch = generate_test_array(shape=(3, 2, 4), dtype="float32")
        vmapped_fn = nb.vmap(identity_fn)
        result = vmapped_fn(x_batch)

        assert_tensor_close(result, x_batch.data)

    def test_vmap_constant_function(self):
        """Test vmap with function that returns constant."""

        def constant_fn(x):
            return nb.Tensor(np.array([[1.0, 2.0]]), dtype="float32")

        x_batch = generate_test_array(shape=(3, 2), dtype="float32")
        vmapped_fn = nb.vmap(constant_fn)
        result = vmapped_fn(x_batch)

        # Should return the same constant for each batch element
        expected = np.tile(np.array([[1.0, 2.0]]), (3, 1, 1))
        assert_tensor_close(result, expected)


class TestVmapGradients:
    """Test gradients through vmap transformations."""

    def test_vmap_gradients_unary(self):
        """Test gradients through vmap with unary operations."""

        def test_fn(x):
            return nb.sin(x)

        x_batch = generate_test_array(
            shape=(3, 2, 2), dtype="float32", requires_grad=True
        )
        vmapped_fn = nb.vmap(test_fn)
        result = vmapped_fn(x_batch)

        loss = nb.reduce_sum(result)
        loss.backward()

        assert x_batch.grad is not None
        assert x_batch.grad.shape == x_batch.shape
        assert x_batch.grad.dtype == x_batch.dtype

    def test_vmap_gradients_binary(self):
        """Test gradients through vmap with binary operations."""

        def test_fn(x, y):
            return nb.add(x, y)

        x_batch = generate_test_array(
            shape=(3, 2, 2), dtype="float32", requires_grad=True
        )
        y_batch = generate_test_array(
            shape=(3, 2, 2), dtype="float32", requires_grad=True
        )

        vmapped_fn = nb.vmap(test_fn)
        result = vmapped_fn(x_batch, y_batch)

        loss = nb.reduce_sum(result)
        loss.backward()

        assert x_batch.grad is not None
        assert y_batch.grad is not None
        assert x_batch.grad.shape == x_batch.shape
        assert y_batch.grad.shape == y_batch.shape

    def test_vmap_nested_gradients(self):
        """Test gradients through nested vmap operations."""

        def inner_fn(x):
            return nb.exp(x)

        def outer_fn(x_batch):
            vmapped_inner = nb.vmap(inner_fn)
            return nb.reduce_sum(vmapped_inner(x_batch))

        x_batch = generate_test_array(
            shape=(3, 2, 2), dtype="float32", requires_grad=True
        )
        result = outer_fn(x_batch)
        result.backward()

        assert x_batch.grad is not None
        assert x_batch.grad.shape == x_batch.shape


class TestVmapCompatibility:
    """Test vmap compatibility with different operations."""

    @pytest.mark.parametrize(
        "op_name", ["negate", "exp", "log", "sin", "cos", "relu", "transpose"]
    )
    def test_vmap_unary_operation_compatibility(self, op_name):
        """Test that all unary operations work with vmap."""

        def create_test_fn(op_name):
            def test_fn(x):
                if op_name == "negate":
                    return nb.negate(x)
                elif op_name == "exp":
                    return nb.exp(x)
                elif op_name == "log":
                    return nb.log(x)
                elif op_name == "sin":
                    return nb.sin(x)
                elif op_name == "cos":
                    return nb.cos(x)
                elif op_name == "relu":
                    return nb.relu(x)
                elif op_name == "transpose":
                    return nb.transpose(x)

            return test_fn

        kwargs = {"ensure_positive": True} if op_name == "log" else {}
        x_batch = generate_test_array(shape=(2, 3, 3), dtype="float32", **kwargs)

        test_fn = create_test_fn(op_name)
        vmapped_fn = nb.vmap(test_fn)

        # Should not raise an exception
        result = vmapped_fn(x_batch)
        assert result is not None
        assert result.shape[0] == 2  # Batch dimension preserved

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div", "power"])
    def test_vmap_binary_operation_compatibility(self, op_name):
        """Test that all binary operations work with vmap."""

        def create_test_fn(op_name):
            def test_fn(x, y):
                if op_name == "add":
                    return nb.add(x, y)
                elif op_name == "sub":
                    return nb.sub(x, y)
                elif op_name == "mul":
                    return nb.mul(x, y)
                elif op_name == "div":
                    return nb.div(x, y)
                elif op_name == "power":
                    return nb.power(x, y)

            return test_fn

        x_kwargs = {"ensure_positive": True} if op_name == "power" else {}
        y_kwargs = {"for_binary_op_rhs": True} if op_name in ["div", "power"] else {}

        x_batch = generate_test_array(shape=(2, 3, 3), dtype="float32", **x_kwargs)
        y_batch = generate_test_array(shape=(2, 3, 3), dtype="float32", **y_kwargs)

        test_fn = create_test_fn(op_name)
        vmapped_fn = nb.vmap(test_fn)

        # Should not raise an exception
        result = vmapped_fn(x_batch, y_batch)
        assert result is not None
        assert result.shape[0] == 2  # Batch dimension preserved
