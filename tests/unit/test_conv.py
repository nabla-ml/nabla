#!/usr/bin/env python3
"""Comprehensive test suite for conv2d and conv2d_transpose operations against JAX implementations."""

import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad

# Add nabla to path
sys.path.insert(0, "/Users/tillife/Documents/CodingProjects/nabla")

from nabla.core.trafos import vjp as nabla_vjp
from nabla.ops.creation import array
from nabla.ops.linalg import conv2d, conv2d_transpose


class ConvTestConfig:
    """Configuration for convolution tests."""

    def __init__(
        self,
        batch_size: int = 1,
        in_channels: int = 3,
        out_channels: int = 4,
        input_height: int = 8,
        input_width: int = 8,
        kernel_size: int = 3,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        dilation: tuple[int, int] = (1, 1),
        output_padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        seed: int = 42,
        name: str = "default",
    ):
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.seed = seed
        self.name = name

    def __repr__(self):
        return f"ConvTestConfig({self.name})"


def generate_test_configs() -> list[ConvTestConfig]:
    """Generate comprehensive test configurations."""
    configs = []

    # Basic configurations
    configs.append(
        ConvTestConfig(
            name="basic_conv",
            batch_size=2,
            in_channels=3,
            out_channels=4,
            input_height=8,
            input_width=8,
            kernel_size=3,
            stride=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )

    # With padding
    configs.append(
        ConvTestConfig(
            name="with_padding",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=6,
            input_width=6,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Asymmetric padding
    configs.append(
        ConvTestConfig(
            name="asymmetric_padding",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=6,
            input_width=6,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 2, 0, 1),
        )
    )

    # Strided convolution
    configs.append(
        ConvTestConfig(
            name="strided_conv",
            batch_size=1,
            in_channels=3,
            out_channels=2,
            input_height=8,
            input_width=8,
            kernel_size=3,
            stride=(2, 2),
            padding=(1, 1, 1, 1),
        )
    )

    # Asymmetric stride
    configs.append(
        ConvTestConfig(
            name="asymmetric_stride",
            batch_size=1,
            in_channels=2,
            out_channels=4,
            input_height=10,
            input_width=8,
            kernel_size=3,
            stride=(2, 1),
            padding=(0, 0, 0, 0),
        )
    )

    # Dilated convolution
    configs.append(
        ConvTestConfig(
            name="dilated_conv",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=10,
            input_width=10,
            kernel_size=3,
            stride=(1, 1),
            padding=(2, 2, 2, 2),
            dilation=(2, 2),
        )
    )

    # Large kernel
    configs.append(
        ConvTestConfig(
            name="large_kernel",
            batch_size=1,
            in_channels=2,
            out_channels=2,
            input_height=12,
            input_width=12,
            kernel_size=5,
            stride=(1, 1),
            padding=(2, 2, 2, 2),
        )
    )

    # 1x1 convolution
    configs.append(
        ConvTestConfig(
            name="1x1_conv",
            batch_size=2,
            in_channels=4,
            out_channels=8,
            input_height=8,
            input_width=8,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )

    # Single channel
    configs.append(
        ConvTestConfig(
            name="single_channel",
            batch_size=1,
            in_channels=1,
            out_channels=1,
            input_height=5,
            input_width=5,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Many channels
    configs.append(
        ConvTestConfig(
            name="many_channels",
            batch_size=1,
            in_channels=16,
            out_channels=32,
            input_height=8,
            input_width=8,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Edge case: minimal size
    configs.append(
        ConvTestConfig(
            name="minimal_size",
            batch_size=1,
            in_channels=1,
            out_channels=1,
            input_height=3,
            input_width=3,
            kernel_size=3,
            stride=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )

    # NEW COMPREHENSIVE CONFIGURATIONS

    # Stress test: Large batch size
    configs.append(
        ConvTestConfig(
            name="large_batch",
            batch_size=8,
            in_channels=4,
            out_channels=6,
            input_height=16,
            input_width=16,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Rectangular input (tall)
    configs.append(
        ConvTestConfig(
            name="tall_input",
            batch_size=2,
            in_channels=3,
            out_channels=5,
            input_height=16,
            input_width=8,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Rectangular input (wide)
    configs.append(
        ConvTestConfig(
            name="wide_input",
            batch_size=2,
            in_channels=3,
            out_channels=5,
            input_height=8,
            input_width=16,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Rectangular kernel
    configs.append(
        ConvTestConfig(
            name="rect_kernel_2x3",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=8,
            input_width=8,
            kernel_size=2,  # This will be overridden in test
            stride=(1, 1),
            padding=(0, 1, 1, 0),
        )
    )

    # Very large stride
    configs.append(
        ConvTestConfig(
            name="very_large_stride",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=20,
            input_width=20,
            kernel_size=3,
            stride=(4, 4),
            padding=(1, 1, 1, 1),
        )
    )

    # Extreme asymmetric stride
    configs.append(
        ConvTestConfig(
            name="extreme_asym_stride",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=16,
            input_width=16,
            kernel_size=3,
            stride=(1, 4),
            padding=(1, 1, 1, 1),
        )
    )

    # High dilation
    configs.append(
        ConvTestConfig(
            name="high_dilation",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=16,
            input_width=16,
            kernel_size=3,
            stride=(1, 1),
            padding=(4, 4, 4, 4),
            dilation=(3, 3),
        )
    )

    # Asymmetric dilation
    configs.append(
        ConvTestConfig(
            name="asym_dilation",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=12,
            input_width=12,
            kernel_size=3,
            stride=(1, 1),
            padding=(2, 3, 1, 4),
            dilation=(2, 3),
        )
    )

    # Large kernel with stride
    configs.append(
        ConvTestConfig(
            name="large_kernel_stride",
            batch_size=1,
            in_channels=3,
            out_channels=4,
            input_height=20,
            input_width=20,
            kernel_size=7,
            stride=(3, 3),
            padding=(3, 3, 3, 3),
        )
    )

    # Very small input with padding
    configs.append(
        ConvTestConfig(
            name="tiny_input_big_pad",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=2,
            input_width=2,
            kernel_size=3,
            stride=(1, 1),
            padding=(2, 2, 2, 2),
        )
    )

    # No padding, kernel larger than half input
    configs.append(
        ConvTestConfig(
            name="big_kernel_no_pad",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=7,
            input_width=7,
            kernel_size=5,
            stride=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )

    # Extreme channel counts
    configs.append(
        ConvTestConfig(
            name="extreme_channels",
            batch_size=1,
            in_channels=64,
            out_channels=128,
            input_height=8,
            input_width=8,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Single pixel input
    configs.append(
        ConvTestConfig(
            name="single_pixel",
            batch_size=1,
            in_channels=3,
            out_channels=5,
            input_height=1,
            input_width=1,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )

    # Complex combination: stride + dilation + asymmetric padding
    configs.append(
        ConvTestConfig(
            name="complex_combo_1",
            batch_size=2,
            in_channels=4,
            out_channels=6,
            input_height=15,
            input_width=12,
            kernel_size=4,
            stride=(2, 3),
            padding=(1, 2, 0, 3),
            dilation=(2, 1),
        )
    )

    # Another complex combination
    configs.append(
        ConvTestConfig(
            name="complex_combo_2",
            batch_size=1,
            in_channels=5,
            out_channels=7,
            input_height=18,
            input_width=14,
            kernel_size=5,
            stride=(3, 2),
            padding=(2, 1, 3, 0),
            dilation=(1, 2),
        )
    )

    # Edge case: kernel size equals input size
    configs.append(
        ConvTestConfig(
            name="kernel_eq_input",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=5,
            input_width=5,
            kernel_size=5,
            stride=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )

    # Edge case: stride larger than kernel
    configs.append(
        ConvTestConfig(
            name="stride_gt_kernel",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=12,
            input_width=12,
            kernel_size=3,
            stride=(5, 5),
            padding=(1, 1, 1, 1),
        )
    )

    # Extreme asymmetric padding
    configs.append(
        ConvTestConfig(
            name="extreme_asym_pad",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=8,
            input_width=8,
            kernel_size=3,
            stride=(1, 1),
            padding=(0, 5, 2, 0),
        )
    )

    # Large input with minimal kernel
    configs.append(
        ConvTestConfig(
            name="large_input_small_kernel",
            batch_size=1,
            in_channels=3,
            out_channels=4,
            input_height=32,
            input_width=32,
            kernel_size=1,
            stride=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )

    # Power-of-2 dimensions (common in practice)
    configs.append(
        ConvTestConfig(
            name="pow2_dims",
            batch_size=4,
            in_channels=8,
            out_channels=16,
            input_height=32,
            input_width=32,
            kernel_size=3,
            stride=(2, 2),
            padding=(1, 1, 1, 1),
        )
    )

    # Odd dimensions everywhere
    configs.append(
        ConvTestConfig(
            name="all_odd_dims",
            batch_size=3,
            in_channels=5,
            out_channels=7,
            input_height=9,
            input_width=11,
            kernel_size=5,
            stride=(3, 3),
            padding=(2, 2, 2, 2),
        )
    )

    # Prime number dimensions (unusual but good stress test)
    configs.append(
        ConvTestConfig(
            name="prime_dims",
            batch_size=1,
            in_channels=7,
            out_channels=11,
            input_height=13,
            input_width=17,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Very high dilation with small kernel
    configs.append(
        ConvTestConfig(
            name="extreme_dilation",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=25,
            input_width=25,
            kernel_size=3,
            stride=(1, 1),
            padding=(8, 8, 8, 8),
            dilation=(5, 5),
        )
    )

    # Gradient stress test: many small operations
    configs.append(
        ConvTestConfig(
            name="gradient_stress",
            batch_size=1,
            in_channels=32,
            out_channels=32,
            input_height=4,
            input_width=4,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )

    # Memory stress test
    configs.append(
        ConvTestConfig(
            name="memory_stress",
            batch_size=2,
            in_channels=16,
            out_channels=24,
            input_height=24,
            input_width=24,
            kernel_size=5,
            stride=(1, 1),
            padding=(2, 2, 2, 2),
        )
    )

    # Boundary condition: exactly fitting convolution
    configs.append(
        ConvTestConfig(
            name="exact_fit",
            batch_size=1,
            in_channels=2,
            out_channels=3,
            input_height=8,
            input_width=8,
            kernel_size=4,
            stride=(4, 4),
            padding=(0, 0, 0, 0),
        )
    )

    # Another boundary: output size = 1
    configs.append(
        ConvTestConfig(
            name="output_size_1",
            batch_size=1,
            in_channels=3,
            out_channels=5,
            input_height=5,
            input_width=5,
            kernel_size=5,
            stride=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )

    return configs


@pytest.mark.skip(reason="ConvTestConfig fixture not available - needs implementation")
@pytest.mark.skip(reason="ConvTestConfig fixture not available - needs implementation")
def test_conv2d_forward(config: ConvTestConfig) -> dict[str, Any]:
    """Test conv2d forward pass for a given configuration."""
    np.random.seed(config.seed)

    # Generate test data
    input_data = np.random.randn(
        config.batch_size, config.input_height, config.input_width, config.in_channels
    ).astype(np.float32)
    filter_data = np.random.randn(
        config.kernel_size, config.kernel_size, config.in_channels, config.out_channels
    ).astype(np.float32)

    try:
        # Nabla computation (NHWC format)
        nabla_input = array(input_data)
        nabla_filter = array(filter_data)
        nabla_result = conv2d(
            nabla_input,
            nabla_filter,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
        )
        nabla_output = nabla_result.to_numpy()

        # JAX computation (needs NCHW format)
        jax_input = jnp.transpose(input_data, (0, 3, 1, 2))  # NHWC -> NCHW
        jax_filter = jnp.transpose(filter_data, (3, 2, 0, 1))  # HWIO -> OIHW

        # Convert padding format to JAX format
        jax_padding = (
            (config.padding[0], config.padding[1]),
            (config.padding[2], config.padding[3]),
        )

        jax_result = jax.lax.conv_general_dilated(
            jax_input,
            jax_filter,
            window_strides=config.stride,
            padding=jax_padding,
            lhs_dilation=(1, 1),
            rhs_dilation=config.dilation,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        # Convert JAX result back to NHWC
        jax_output = np.transpose(jax_result, (0, 2, 3, 1))

        # Calculate differences
        max_diff = np.max(np.abs(nabla_output - jax_output))
        mean_diff = np.mean(np.abs(nabla_output - jax_output))
        is_close = np.allclose(nabla_output, jax_output, rtol=1e-5, atol=1e-6)

        return {
            "config": config,
            "success": True,
            "nabla_shape": nabla_output.shape,
            "jax_shape": jax_output.shape,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "is_close": is_close,
            "test_data": (input_data, filter_data, nabla_output, jax_output),
            "error": None,
        }

    except Exception as e:
        return {
            "config": config,
            "success": False,
            "nabla_shape": "N/A",
            "jax_shape": "N/A",
            "max_diff": float("inf"),
            "mean_diff": float("inf"),
            "is_close": False,
            "test_data": None,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@pytest.mark.skip(reason="ConvTestConfig fixture not available - needs implementation")
def test_conv2d_gradients(config: ConvTestConfig) -> dict[str, Any]:
    """Test conv2d gradients (VJP) for a given configuration."""
    np.random.seed(config.seed + 1000)  # Different seed for gradient tests

    try:
        # Generate test data
        input_data = np.random.randn(
            config.batch_size,
            config.input_height,
            config.input_width,
            config.in_channels,
        ).astype(np.float32)
        filter_data = np.random.randn(
            config.kernel_size,
            config.kernel_size,
            config.in_channels,
            config.out_channels,
        ).astype(np.float32)

        # JAX reference functions
        def jax_conv_func(x):
            x_nchw = jnp.transpose(x, (0, 3, 1, 2))
            filter_oihw = jnp.transpose(filter_data, (3, 2, 0, 1))
            result = jax.lax.conv_general_dilated(
                x_nchw,
                filter_oihw,
                window_strides=config.stride,
                padding=(
                    (config.padding[0], config.padding[1]),
                    (config.padding[2], config.padding[3]),
                ),
                lhs_dilation=(1, 1),
                rhs_dilation=config.dilation,
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            return jnp.transpose(result, (0, 2, 3, 1))

        def jax_conv_func_filter(w):
            x_nchw = jnp.transpose(input_data, (0, 3, 1, 2))
            filter_oihw = jnp.transpose(w, (3, 2, 0, 1))
            result = jax.lax.conv_general_dilated(
                x_nchw,
                filter_oihw,
                window_strides=config.stride,
                padding=(
                    (config.padding[0], config.padding[1]),
                    (config.padding[2], config.padding[3]),
                ),
                lhs_dilation=(1, 1),
                rhs_dilation=config.dilation,
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            return jnp.transpose(result, (0, 2, 3, 1))

        # Compute JAX reference gradients using automatic differentiation
        jax_input_grad = grad(lambda x: jnp.sum(jax_conv_func(x)))(input_data)
        jax_filter_grad = grad(lambda w: jnp.sum(jax_conv_func_filter(w)))(filter_data)

        # Get output shape for cotangent
        output_shape = jax_conv_func(input_data).shape
        cotangent = array(np.ones(output_shape, dtype=np.float32))

        # Nabla VJP functions
        def nabla_loss_func(x):
            filter_arr = array(filter_data)
            result = conv2d(
                x,
                filter_arr,
                stride=config.stride,
                padding=config.padding,
                dilation=config.dilation,
            )
            return result

        def nabla_loss_func_filter(w):
            x_arr = array(input_data)
            result = conv2d(
                x_arr,
                w,
                stride=config.stride,
                padding=config.padding,
                dilation=config.dilation,
            )
            return result

        # Compute Nabla gradients using VJP
        input_grad_success = False
        filter_grad_success = False
        input_grad_error = None
        filter_grad_error = None
        nabla_input_grad = None
        nabla_filter_grad = None

        # Test input gradient
        try:
            _, vjp_func = nabla_vjp(nabla_loss_func, array(input_data))
            nabla_input_grad = vjp_func(cotangent)[0].to_numpy()
            input_grad_success = True
        except Exception as e:
            input_grad_error = f"{type(e).__name__}: {str(e)}"

        # Test filter gradient
        try:
            _, vjp_func_filter = nabla_vjp(nabla_loss_func_filter, array(filter_data))
            nabla_filter_grad = vjp_func_filter(cotangent)[0].to_numpy()
            filter_grad_success = True
        except Exception as e:
            filter_grad_error = f"{type(e).__name__}: {str(e)}"

        # Compare gradients
        results = {
            "config": config,
            "input_grad_success": input_grad_success,
            "filter_grad_success": filter_grad_success,
            "input_grad_error": input_grad_error,
            "filter_grad_error": filter_grad_error,
            "detailed_info": {
                "input_shape": input_data.shape,
                "filter_shape": filter_data.shape,
                "output_shape": output_shape,
                "stride": config.stride,
                "padding": config.padding,
                "dilation": config.dilation,
                "jax_input_grad_shape": jax_input_grad.shape,
                "jax_filter_grad_shape": jax_filter_grad.shape,
            },
        }

        if input_grad_success:
            input_max_diff = np.max(np.abs(nabla_input_grad - jax_input_grad))
            input_mean_diff = np.mean(np.abs(nabla_input_grad - jax_input_grad))
            input_is_close = np.allclose(
                nabla_input_grad, jax_input_grad, rtol=1e-4, atol=1e-5
            )

            results.update(
                {
                    "input_grad_max_diff": input_max_diff,
                    "input_grad_mean_diff": input_mean_diff,
                    "input_grad_is_close": input_is_close,
                    "nabla_input_grad_shape": nabla_input_grad.shape,
                    "jax_input_grad_shape": jax_input_grad.shape,
                }
            )

        if filter_grad_success:
            filter_max_diff = np.max(np.abs(nabla_filter_grad - jax_filter_grad))
            filter_mean_diff = np.mean(np.abs(nabla_filter_grad - jax_filter_grad))
            filter_is_close = np.allclose(
                nabla_filter_grad, jax_filter_grad, rtol=1e-4, atol=1e-5
            )

            results.update(
                {
                    "filter_grad_max_diff": filter_max_diff,
                    "filter_grad_mean_diff": filter_mean_diff,
                    "filter_grad_is_close": filter_is_close,
                    "nabla_filter_grad_shape": nabla_filter_grad.shape,
                    "jax_filter_grad_shape": jax_filter_grad.shape,
                }
            )

        return results

    except Exception as e:
        # High-level error in test setup
        return {
            "config": config,
            "input_grad_success": False,
            "filter_grad_success": False,
            "input_grad_error": f"Test setup error: {type(e).__name__}: {str(e)}",
            "filter_grad_error": f"Test setup error: {type(e).__name__}: {str(e)}",
            "error": str(e),
            "error_type": type(e).__name__,
            "detailed_info": {
                "input_shape": "N/A",
                "filter_shape": "N/A",
                "output_shape": "N/A",
                "stride": config.stride,
                "padding": config.padding,
                "dilation": config.dilation,
            },
        }


@pytest.mark.skip(reason="ConvTestConfig fixture not available - needs implementation")
def test_conv2d_transpose_forward(config: ConvTestConfig) -> dict[str, Any]:
    """Test conv2d_transpose forward pass for a given configuration."""
    np.random.seed(config.seed + 2000)

    # For transpose conv, we need different input/output channel relationship
    # Filter shape for transpose conv: (K_H, K_W, C_out, C_in) in HWOI format
    input_data = np.random.randn(
        config.batch_size, config.input_height, config.input_width, config.in_channels
    ).astype(np.float32)
    filter_data = np.random.randn(
        config.kernel_size, config.kernel_size, config.out_channels, config.in_channels
    ).astype(np.float32)

    try:
        # Nabla computation (NHWC format)
        nabla_input = array(input_data)
        nabla_filter = array(filter_data)
        nabla_result = conv2d_transpose(
            nabla_input,
            nabla_filter,
            stride=config.stride,
            padding=config.padding,
            output_padding=config.output_padding,
            dilation=config.dilation,
        )
        nabla_output = nabla_result.to_numpy()

        # JAX computation (needs NCHW format)
        jax_input = jnp.transpose(input_data, (0, 3, 1, 2))  # NHWC -> NCHW
        # For JAX transpose_conv: filter should be IOHW format
        jax_filter = jnp.transpose(filter_data, (3, 2, 0, 1))  # HWOI -> IOHW

        # Convert padding format to JAX format
        jax_padding = (
            (config.padding[0], config.padding[1]),
            (config.padding[2], config.padding[3]),
        )

        jax_result = jax.lax.conv_transpose(
            jax_input,
            jax_filter,
            strides=config.stride,
            padding=jax_padding,
            rhs_dilation=config.dilation,
            dimension_numbers=("NCHW", "IOHW", "NCHW"),
        )

        # Convert JAX result back to NHWC
        jax_output = np.transpose(jax_result, (0, 2, 3, 1))

        # Calculate differences
        max_diff = np.max(np.abs(nabla_output - jax_output))
        mean_diff = np.mean(np.abs(nabla_output - jax_output))
        is_close = np.allclose(nabla_output, jax_output, rtol=1e-5, atol=1e-6)

        return {
            "config": config,
            "success": True,
            "nabla_shape": nabla_output.shape,
            "jax_shape": jax_output.shape,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "is_close": is_close,
            "test_data": (input_data, filter_data, nabla_output, jax_output),
            "error": None,
            "detailed_info": {
                "input_shape": input_data.shape,
                "filter_shape": filter_data.shape,
                "stride": config.stride,
                "padding": config.padding,
                "dilation": config.dilation,
                "output_padding": config.output_padding,
            },
        }

    except Exception as e:
        # Calculate expected output shape for diagnostics
        try:
            # Try to compute expected shape using JAX
            jax_input = jnp.transpose(input_data, (0, 3, 1, 2))
            jax_filter = jnp.transpose(filter_data, (3, 2, 0, 1))
            jax_padding = (
                (config.padding[0], config.padding[1]),
                (config.padding[2], config.padding[3]),
            )

            jax_result = jax.lax.conv_transpose(
                jax_input,
                jax_filter,
                strides=config.stride,
                padding=jax_padding,
                rhs_dilation=config.dilation,
                dimension_numbers=("NCHW", "IOHW", "NCHW"),
            )
            expected_shape = np.transpose(jax_result, (0, 2, 3, 1)).shape
        except:
            expected_shape = "Could not compute"

        return {
            "config": config,
            "success": False,
            "nabla_shape": "ERROR",
            "jax_shape": expected_shape,
            "max_diff": float("inf"),
            "mean_diff": float("inf"),
            "is_close": False,
            "test_data": None,
            "error": str(e),
            "error_type": type(e).__name__,
            "detailed_info": {
                "input_shape": input_data.shape,
                "filter_shape": filter_data.shape,
                "stride": config.stride,
                "padding": config.padding,
                "dilation": config.dilation,
                "output_padding": config.output_padding,
                "expected_jax_shape": expected_shape,
            },
        }


@pytest.mark.skip(reason="ConvTestConfig fixture not available - needs implementation")
def test_conv2d_transpose_gradients(config: ConvTestConfig) -> dict[str, Any]:
    """Test conv2d_transpose gradients (VJP) for a given configuration."""
    np.random.seed(config.seed + 3000)

    try:
        # Generate test data for transpose conv
        input_data = np.random.randn(
            config.batch_size,
            config.input_height,
            config.input_width,
            config.in_channels,
        ).astype(np.float32)
        # Filter shape for transpose conv: (K_H, K_W, C_out, C_in) HWOI
        filter_data = np.random.randn(
            config.kernel_size,
            config.kernel_size,
            config.out_channels,
            config.in_channels,
        ).astype(np.float32)

        # JAX reference functions
        def jax_conv_transpose_func(x):
            x_nchw = jnp.transpose(x, (0, 3, 1, 2))
            filter_iohw = jnp.transpose(filter_data, (3, 2, 0, 1))
            result = jax.lax.conv_transpose(
                x_nchw,
                filter_iohw,
                strides=config.stride,
                padding=(
                    (config.padding[0], config.padding[1]),
                    (config.padding[2], config.padding[3]),
                ),
                rhs_dilation=config.dilation,
                dimension_numbers=("NCHW", "IOHW", "NCHW"),
            )
            return jnp.transpose(result, (0, 2, 3, 1))

        def jax_conv_transpose_func_filter(w):
            x_nchw = jnp.transpose(input_data, (0, 3, 1, 2))
            filter_iohw = jnp.transpose(w, (3, 2, 0, 1))
            result = jax.lax.conv_transpose(
                x_nchw,
                filter_iohw,
                strides=config.stride,
                padding=(
                    (config.padding[0], config.padding[1]),
                    (config.padding[2], config.padding[3]),
                ),
                rhs_dilation=config.dilation,
                dimension_numbers=("NCHW", "IOHW", "NCHW"),
            )
            return jnp.transpose(result, (0, 2, 3, 1))

        # Compute JAX reference gradients using automatic differentiation
        jax_input_grad = grad(lambda x: jnp.sum(jax_conv_transpose_func(x)))(input_data)
        jax_filter_grad = grad(lambda w: jnp.sum(jax_conv_transpose_func_filter(w)))(
            filter_data
        )

        # Get output shape for cotangent
        try:
            output_shape = jax_conv_transpose_func(input_data).shape
            cotangent = array(np.ones(output_shape, dtype=np.float32))
        except Exception as e:
            return {
                "config": config,
                "input_grad_success": False,
                "filter_grad_success": False,
                "input_grad_error": f"JAX output shape computation failed: {type(e).__name__}: {str(e)}",
                "filter_grad_error": f"JAX output shape computation failed: {type(e).__name__}: {str(e)}",
                "error": f"JAX forward pass failed: {str(e)}",
                "error_type": type(e).__name__,
                "detailed_info": {
                    "input_shape": input_data.shape,
                    "filter_shape": filter_data.shape,
                    "output_shape": "Failed to compute",
                    "stride": config.stride,
                    "padding": config.padding,
                    "dilation": config.dilation,
                    "output_padding": config.output_padding,
                },
            }

        # Nabla VJP functions
        def nabla_loss_func(x):
            filter_arr = array(filter_data)
            result = conv2d_transpose(
                x,
                filter_arr,
                stride=config.stride,
                padding=config.padding,
                output_padding=config.output_padding,
                dilation=config.dilation,
            )
            return result

        def nabla_loss_func_filter(w):
            x_arr = array(input_data)
            result = conv2d_transpose(
                x_arr,
                w,
                stride=config.stride,
                padding=config.padding,
                output_padding=config.output_padding,
                dilation=config.dilation,
            )
            return result

        # Compute Nabla gradients using VJP
        input_grad_success = False
        filter_grad_success = False
        input_grad_error = None
        filter_grad_error = None
        nabla_input_grad = None
        nabla_filter_grad = None

        # Test input gradient
        try:
            _, vjp_func = nabla_vjp(nabla_loss_func, array(input_data))
            nabla_input_grad = vjp_func(cotangent)[0].to_numpy()
            input_grad_success = True
        except Exception as e:
            input_grad_error = f"{type(e).__name__}: {str(e)}"

        # Test filter gradient
        try:
            _, vjp_func_filter = nabla_vjp(nabla_loss_func_filter, array(filter_data))
            nabla_filter_grad = vjp_func_filter(cotangent)[0].to_numpy()
            filter_grad_success = True
        except Exception as e:
            filter_grad_error = f"{type(e).__name__}: {str(e)}"

        # Compare gradients
        results = {
            "config": config,
            "input_grad_success": input_grad_success,
            "filter_grad_success": filter_grad_success,
            "input_grad_error": input_grad_error,
            "filter_grad_error": filter_grad_error,
            "detailed_info": {
                "input_shape": input_data.shape,
                "filter_shape": filter_data.shape,
                "output_shape": output_shape,
                "stride": config.stride,
                "padding": config.padding,
                "dilation": config.dilation,
                "output_padding": config.output_padding,
                "jax_input_grad_shape": jax_input_grad.shape,
                "jax_filter_grad_shape": jax_filter_grad.shape,
            },
        }

        if input_grad_success:
            input_max_diff = np.max(np.abs(nabla_input_grad - jax_input_grad))
            input_mean_diff = np.mean(np.abs(nabla_input_grad - jax_input_grad))
            input_is_close = np.allclose(
                nabla_input_grad, jax_input_grad, rtol=1e-4, atol=1e-5
            )

            results.update(
                {
                    "input_grad_max_diff": input_max_diff,
                    "input_grad_mean_diff": input_mean_diff,
                    "input_grad_is_close": input_is_close,
                    "nabla_input_grad_shape": nabla_input_grad.shape,
                    "jax_input_grad_shape": jax_input_grad.shape,
                }
            )

        if filter_grad_success:
            filter_max_diff = np.max(np.abs(nabla_filter_grad - jax_filter_grad))
            filter_mean_diff = np.mean(np.abs(nabla_filter_grad - jax_filter_grad))
            filter_is_close = np.allclose(
                nabla_filter_grad, jax_filter_grad, rtol=1e-4, atol=1e-5
            )

            results.update(
                {
                    "filter_grad_max_diff": filter_max_diff,
                    "filter_grad_mean_diff": filter_mean_diff,
                    "filter_grad_is_close": filter_is_close,
                    "nabla_filter_grad_shape": nabla_filter_grad.shape,
                    "jax_filter_grad_shape": jax_filter_grad.shape,
                }
            )

        return results

    except Exception as e:
        # High-level error in test setup
        return {
            "config": config,
            "input_grad_success": False,
            "filter_grad_success": False,
            "input_grad_error": f"Test setup error: {type(e).__name__}: {str(e)}",
            "filter_grad_error": f"Test setup error: {type(e).__name__}: {str(e)}",
            "error": str(e),
            "error_type": type(e).__name__,
            "detailed_info": {
                "input_shape": "N/A",
                "filter_shape": "N/A",
                "output_shape": "N/A",
                "stride": config.stride,
                "padding": config.padding,
                "dilation": config.dilation,
                "output_padding": config.output_padding,
            },
        }


def print_test_summary(results: list[dict[str, Any]], test_name: str) -> None:
    """Print a summary of test results."""
    print(f"\n{test_name} Results:")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = 0
    failed_tests = []

    for result in results:
        config = result["config"]

        if test_name.endswith("Forward"):
            success = result.get("success", False) and result.get("is_close", False)
            if success:
                passed_tests += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                failed_tests.append(config.name)

            print(
                f"{config.name:20s} | {status} | "
                f"Max diff: {result.get('max_diff', 'N/A'):.2e} | "
                f"Mean diff: {result.get('mean_diff', 'N/A'):.2e} | "
                f"Shape: {result.get('nabla_shape', 'N/A')}"
            )

            # Print error details for failed tests
            if not success and result.get("error"):
                print(
                    f"                     | Error: {result.get('error_type', 'Error')}: {result.get('error')}"
                )

        else:  # Gradient tests
            input_success = result.get("input_grad_success", False) and result.get(
                "input_grad_is_close", False
            )
            filter_success = result.get("filter_grad_success", False) and result.get(
                "filter_grad_is_close", False
            )
            overall_success = input_success and filter_success

            if overall_success:
                passed_tests += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                failed_tests.append(config.name)

            input_status = "✓" if input_success else "✗"
            filter_status = "✓" if filter_success else "✗"

            input_diff = result.get("input_grad_max_diff", float("inf"))
            filter_diff = result.get("filter_grad_max_diff", float("inf"))

            print(
                f"{config.name:20s} | {status} | "
                f"Input: {input_status} ({input_diff:.2e}) | "
                f"Filter: {filter_status} ({filter_diff:.2e})"
            )

            # Print errors if any with more details
            if result.get("input_grad_error"):
                print(
                    f"                     | Input grad error: {result['input_grad_error']}"
                )

            if result.get("filter_grad_error"):
                print(
                    f"                     | Filter grad error: {result['filter_grad_error']}"
                )

            # Print general error if present
            if result.get("error"):
                print(
                    f"                     | General error: {result.get('error_type', 'Error')}: {result.get('error')}"
                )

    print("-" * 80)
    print(f"Summary: {passed_tests}/{total_tests} tests passed")

    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")

    print()


def run_comprehensive_tests():
    """Run comprehensive convolution tests."""
    print("🧪 Comprehensive Conv2D Test Suite")
    print("=" * 80)
    print("Testing Nabla conv2d operations against JAX reference implementation")
    print("This includes various configurations of stride, padding, dilation, etc.")
    print()

    # Generate test configurations
    configs = generate_test_configs()
    print(f"Generated {len(configs)} test configurations")

    # Test conv2d forward pass
    print("\n🔄 Testing conv2d forward pass...")
    conv2d_forward_results = []
    for i, config in enumerate(configs):
        print(f"  Running {config.name} ({i + 1}/{len(configs)})...")
        try:
            result = test_conv2d_forward(config)
            conv2d_forward_results.append(result)
        except Exception as e:
            print(f"❌ {config.name} forward test failed: {e}")
            conv2d_forward_results.append(
                {
                    "config": config,
                    "success": False,
                    "is_close": False,
                    "max_diff": float("inf"),
                    "mean_diff": float("inf"),
                    "nabla_shape": "N/A",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

    print_test_summary(conv2d_forward_results, "Conv2D Forward")

    # Test conv2d gradients
    print("\n🔄 Testing conv2d gradients (VJP)...")
    conv2d_grad_results = []
    for i, config in enumerate(configs):
        print(f"  Running {config.name} gradients ({i + 1}/{len(configs)})...")
        try:
            result = test_conv2d_gradients(config)
            conv2d_grad_results.append(result)
        except Exception as e:
            print(f"❌ {config.name} gradient test failed: {e}")
            conv2d_grad_results.append(
                {
                    "config": config,
                    "input_grad_success": False,
                    "filter_grad_success": False,
                    "input_grad_error": str(e),
                    "filter_grad_error": str(e),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

    print_test_summary(conv2d_grad_results, "Conv2D Gradients")

    # Test conv2d_transpose forward pass
    print("\n🔄 Testing conv2d_transpose forward pass...")
    conv2d_transpose_forward_results = []
    for i, config in enumerate(configs):
        print(f"  Running {config.name} transpose forward ({i + 1}/{len(configs)})...")
        try:
            result = test_conv2d_transpose_forward(config)
            conv2d_transpose_forward_results.append(result)
        except Exception as e:
            print(f"❌ {config.name} transpose forward test failed: {e}")
            conv2d_transpose_forward_results.append(
                {
                    "config": config,
                    "success": False,
                    "is_close": False,
                    "max_diff": float("inf"),
                    "mean_diff": float("inf"),
                    "nabla_shape": "N/A",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

    print_test_summary(conv2d_transpose_forward_results, "Conv2D Transpose Forward")

    # Test conv2d_transpose gradients
    print("\n🔄 Testing conv2d_transpose gradients (VJP)...")
    conv2d_transpose_grad_results = []
    for i, config in enumerate(configs):
        print(
            f"  Running {config.name} transpose gradients ({i + 1}/{len(configs)})..."
        )
        try:
            result = test_conv2d_transpose_gradients(config)
            conv2d_transpose_grad_results.append(result)
        except Exception as e:
            print(f"❌ {config.name} transpose gradient test failed: {e}")
            conv2d_transpose_grad_results.append(
                {
                    "config": config,
                    "input_grad_success": False,
                    "filter_grad_success": False,
                    "input_grad_error": str(e),
                    "filter_grad_error": str(e),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

    print_test_summary(conv2d_transpose_grad_results, "Conv2D Transpose Gradients")

    # Overall summary - Fixed logic here
    total_forward_passed = sum(
        1
        for r in conv2d_forward_results
        if r.get("success", False) and r.get("is_close", False)
    )
    total_transpose_forward_passed = sum(
        1
        for r in conv2d_transpose_forward_results
        if r.get("success", False) and r.get("is_close", False)
    )
    total_grad_passed = sum(
        1
        for r in conv2d_grad_results
        if r.get("input_grad_success", False)
        and r.get("input_grad_is_close", False)
        and r.get("filter_grad_success", False)
        and r.get("filter_grad_is_close", False)
    )
    total_transpose_grad_passed = sum(
        1
        for r in conv2d_transpose_grad_results
        if r.get("input_grad_success", False)
        and r.get("input_grad_is_close", False)
        and r.get("filter_grad_success", False)
        and r.get("filter_grad_is_close", False)
    )

    print("🎯 OVERALL SUMMARY")
    print("=" * 80)
    print(f"Conv2D Forward:           {total_forward_passed}/{len(configs)} passed")
    print(f"Conv2D Gradients:         {total_grad_passed}/{len(configs)} passed")
    print(
        f"Conv2D Transpose Forward: {total_transpose_forward_passed}/{len(configs)} passed"
    )
    print(
        f"Conv2D Transpose Grads:   {total_transpose_grad_passed}/{len(configs)} passed"
    )
    print()

    total_tests = len(configs) * 4
    total_passed = (
        total_forward_passed
        + total_grad_passed
        + total_transpose_forward_passed
        + total_transpose_grad_passed
    )
    success_rate = (total_passed / total_tests) * 100

    print(f"Overall Success Rate: {total_passed}/{total_tests} ({success_rate:.1f}%)")

    if success_rate == 100.0:
        print("🎉 All tests passed! Nabla conv2d operations match JAX perfectly.")
    elif success_rate >= 95.0:
        print("✅ Excellent! Nearly all tests passed.")
    elif success_rate >= 80.0:
        print("⚠️  Good, but some tests failed. Check the details above.")
    else:
        print(
            "❌ Many tests failed. There may be significant issues with the implementation."
        )

    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_tests()
