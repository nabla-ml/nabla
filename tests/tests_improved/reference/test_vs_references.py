# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Reference Implementation Tests
# ===----------------------------------------------------------------------=== #

"""Tests comparing Nabla against reference implementations (NumPy, JAX, PyTorch)."""

import numpy as np
import pytest
from tests_improved.utils.assertions import assert_arrays_close, get_tolerance_for_dtype
from tests_improved.utils.data_generators import (
    SMALL_SHAPES,
    generate_test_data_numpy,
)
from tests_improved.utils.fixtures import (
    JAX_AVAILABLE,
    TORCH_AVAILABLE,
    requires_jax,
    requires_torch,
)

import nabla as nb

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

if TORCH_AVAILABLE:
    import torch


@pytest.mark.reference
class TestAgainstNumPy:
    """Test Nabla operations against NumPy reference implementations."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", SMALL_SHAPES)
    def test_unary_ops_vs_numpy(self, dtype, shape):
        """Test unary operations against NumPy."""
        x_np = generate_test_data_numpy(shape, dtype, low=-2, high=2, seed=42)
        x_nb = nb.Array.from_numpy(x_np)

        rtol, atol = get_tolerance_for_dtype(dtype)

        # Test multiple unary operations
        operations = [
            ("sin", nb.sin, np.sin),
            ("cos", nb.cos, np.cos),
            ("exp", nb.exp, np.exp),
            ("log", nb.log, np.log),  # Note: need positive inputs for log
        ]

        for op_name, nb_op, np_op in operations:
            if op_name == "log":
                # Use positive inputs for logarithm
                x_pos_np = generate_test_data_numpy(
                    shape, dtype, low=0.1, high=5, seed=42
                )
                x_pos_nb = nb.Array.from_numpy(x_pos_np)
                result_nb = nb_op(x_pos_nb)
                expected_np = np_op(x_pos_np)
            else:
                result_nb = nb_op(x_nb)
                expected_np = np_op(x_np)

            assert_arrays_close(
                result_nb,
                expected_np,
                rtol,
                atol,
                f"{op_name} operation mismatch vs NumPy",
            )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", SMALL_SHAPES)
    def test_binary_ops_vs_numpy(self, dtype, shape):
        """Test binary operations against NumPy."""
        a_np = generate_test_data_numpy(shape, dtype, seed=42)
        b_np = generate_test_data_numpy(shape, dtype, seed=43)

        a_nb = nb.Array.from_numpy(a_np)
        b_nb = nb.Array.from_numpy(b_np)

        rtol, atol = get_tolerance_for_dtype(dtype)

        # Test binary operations
        operations = [
            ("add", nb.add, np.add),
            ("sub", nb.sub, np.subtract),
            ("mul", nb.mul, np.multiply),
            ("div", nb.div, np.divide),
        ]

        for op_name, nb_op, np_op in operations:
            if op_name == "div":
                # Avoid division by zero
                b_safe_np = generate_test_data_numpy(
                    shape, dtype, avoid_zeros=True, seed=43
                )
                b_safe_nb = nb.Array.from_numpy(b_safe_np)
                result_nb = nb_op(a_nb, b_safe_nb)
                expected_np = np_op(a_np, b_safe_np)
            else:
                result_nb = nb_op(a_nb, b_nb)
                expected_np = np_op(a_np, b_np)

            assert_arrays_close(
                result_nb,
                expected_np,
                rtol,
                atol,
                f"{op_name} operation mismatch vs NumPy",
            )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_vs_numpy(self, dtype):
        """Test matrix multiplication against NumPy."""
        a_np = generate_test_data_numpy((3, 4), dtype, seed=42)
        b_np = generate_test_data_numpy((4, 5), dtype, seed=43)

        a_nb = nb.Array.from_numpy(a_np)
        b_nb = nb.Array.from_numpy(b_np)

        result_nb = nb.matmul(a_nb, b_nb)
        expected_np = np.matmul(a_np, b_np)

        rtol, atol = get_tolerance_for_dtype(dtype)
        assert_arrays_close(
            result_nb,
            expected_np,
            rtol,
            atol,
            "Matrix multiplication mismatch vs NumPy",
        )


@pytest.mark.reference
@requires_jax
class TestAgainstJAX:
    """Test Nabla operations against JAX reference implementations."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", SMALL_SHAPES[:3])  # Subset for speed
    def test_forward_ops_vs_jax(self, dtype, shape):
        """Test forward pass operations against JAX."""
        x_np = generate_test_data_numpy(shape, dtype, low=-2, high=2, seed=42)

        x_nb = nb.Array.from_numpy(x_np)
        x_jax = jnp.array(x_np)

        rtol, atol = get_tolerance_for_dtype(dtype)

        # Test unary operations
        unary_ops = [
            ("sin", nb.sin, jnp.sin),
            ("cos", nb.cos, jnp.cos),
            ("exp", nb.exp, jnp.exp),
        ]

        for op_name, nb_op, jax_op in unary_ops:
            result_nb = nb_op(x_nb)
            result_jax = jax_op(x_jax)

            assert_arrays_close(
                result_nb,
                result_jax,
                rtol,
                atol,
                f"{op_name} forward pass mismatch vs JAX",
            )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_gradients_vs_jax(self, dtype):
        """Test gradients against JAX autodiff."""
        # Simple test case
        shape = (2, 3)
        x_np = generate_test_data_numpy(shape, dtype, seed=42)

        # Test VJP for a simple function: f(x) = sum(sin(x))
        x_nb = nb.Array.from_numpy(x_np)
        x_jax = jnp.array(x_np)

        # Create cotangent (gradient of loss w.r.t. output)
        cotangent_np = np.array(1.0, dtype=dtype)  # Scalar cotangent for sum
        cotangent_nb = nb.Array.from_numpy(cotangent_np)

        # Nabla VJP
        def nabla_fn(inputs):
            x = inputs[0]
            sin_x = nb.sin(x)
            return [nb.sum(sin_x)]

        output_nb, vjp_fn_nb = nb.vjp(nabla_fn, [x_nb])
        grads_nb = vjp_fn_nb([cotangent_nb])

        # JAX VJP
        def jax_fn(x):
            return jnp.sum(jnp.sin(x))

        output_jax, vjp_fn_jax = jax.vjp(jax_fn, x_jax)
        grad_jax = vjp_fn_jax(cotangent_np)[0]

        # Compare gradients
        rtol, atol = get_tolerance_for_dtype(dtype, is_gradient=True)
        assert_arrays_close(
            grads_nb[0], grad_jax, rtol, atol, "VJP gradient mismatch vs JAX"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_gradients_vs_jax(self, dtype):
        """Test matrix multiplication gradients against JAX."""
        shape_a, shape_b = (2, 3), (3, 2)

        a_np = generate_test_data_numpy(shape_a, dtype, seed=42)
        b_np = generate_test_data_numpy(shape_b, dtype, seed=43)

        # Setup for VJP test
        primals_nb = [nb.Array.from_numpy(a_np), nb.Array.from_numpy(b_np)]
        primals_jax = [jnp.array(a_np), jnp.array(b_np)]

        # Create cotangent for matmul output
        output_shape = (shape_a[0], shape_b[1])
        cotangent_np = generate_test_data_numpy(output_shape, dtype, seed=44)
        cotangent_nb = nb.Array.from_numpy(cotangent_np)

        # Nabla VJP
        nabla_op = lambda inputs: [nb.matmul(inputs[0], inputs[1])]
        _, vjp_fn_nb = nb.vjp(nabla_op, primals_nb)
        grads_nb = vjp_fn_nb([cotangent_nb])

        # JAX VJP
        jax_op = lambda a, b: jnp.matmul(a, b)
        _, vjp_fn_jax = jax.vjp(jax_op, *primals_jax)
        grads_jax = vjp_fn_jax(cotangent_np)

        # Compare gradients
        rtol, atol = get_tolerance_for_dtype(dtype, is_gradient=True)
        for i, (grad_nb, grad_jax) in enumerate(zip(grads_nb, grads_jax, strict=False)):
            assert_arrays_close(
                grad_nb, grad_jax, rtol, atol, f"Matmul gradient {i} mismatch vs JAX"
            )


@pytest.mark.reference
@requires_torch
class TestAgainstPyTorch:
    """Test Nabla operations against PyTorch reference implementations."""

    @pytest.mark.parametrize(
        "dtype_np,dtype_torch",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_forward_ops_vs_pytorch(self, dtype_np, dtype_torch):
        """Test forward operations against PyTorch."""
        shape = (2, 3)
        x_np = generate_test_data_numpy(shape, dtype_np, low=-2, high=2, seed=42)

        x_nb = nb.Array.from_numpy(x_np)
        x_torch = torch.tensor(x_np, dtype=dtype_torch)

        rtol, atol = get_tolerance_for_dtype(dtype_np)

        # Test operations that exist in both
        operations = [
            ("sin", nb.sin, torch.sin),
            ("cos", nb.cos, torch.cos),
            ("exp", nb.exp, torch.exp),
        ]

        for op_name, nb_op, torch_op in operations:
            result_nb = nb_op(x_nb)
            result_torch = torch_op(x_torch)

            # Convert PyTorch result to numpy
            result_torch_np = result_torch.detach().numpy()

            assert_arrays_close(
                result_nb,
                result_torch_np,
                rtol,
                atol,
                f"{op_name} forward pass mismatch vs PyTorch",
            )

    @pytest.mark.parametrize(
        "dtype_np,dtype_torch",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_matmul_vs_pytorch(self, dtype_np, dtype_torch):
        """Test matrix multiplication against PyTorch."""
        a_np = generate_test_data_numpy((2, 3), dtype_np, seed=42)
        b_np = generate_test_data_numpy((3, 4), dtype_np, seed=43)

        # Nabla
        a_nb = nb.Array.from_numpy(a_np)
        b_nb = nb.Array.from_numpy(b_np)
        result_nb = nb.matmul(a_nb, b_nb)

        # PyTorch
        a_torch = torch.tensor(a_np, dtype=dtype_torch)
        b_torch = torch.tensor(b_np, dtype=dtype_torch)
        result_torch = torch.matmul(a_torch, b_torch)

        # Compare
        rtol, atol = get_tolerance_for_dtype(dtype_np)
        assert_arrays_close(
            result_nb,
            result_torch.detach().numpy(),
            rtol,
            atol,
            "Matrix multiplication mismatch vs PyTorch",
        )
