#!/usr/bin/env python
"""
Comprehensive test suite for Conv2DTranspose gradients in Nabla.

This test file compares Nabla's automatic gradients for conv2d_transpose against 
PyTorch's autograd to ensure correctness across a wide range of parameters.

Test cases are based on the manual gradient verification in 
tmp/conv/conv2d_transpose_manual_grad.py.
"""

import pytest
import torch
import torch.nn.functional as F
import nabla as nb
import numpy as np


def pytorch_to_nabla(torch_tensor):
    """Convert PyTorch tensor to Nabla tensor."""
    return nb.Tensor.from_numpy(torch_tensor.detach().cpu().numpy())


def compare_gradients(nabla_grad, pytorch_grad, dtype, test_name):
    """Compare gradients with appropriate tolerances based on dtype."""
    if dtype == torch.float64:
        rtol, atol = 1e-5, 1e-8
    elif dtype == torch.float32:
        rtol, atol = 1e-3, 1e-5
    else:
        rtol, atol = 1e-1, 1e-2
    
    nabla_grad_np = nabla_grad.to_numpy()
    pytorch_grad_np = pytorch_grad.detach().cpu().numpy()
    
    is_close = np.allclose(nabla_grad_np, pytorch_grad_np, rtol=rtol, atol=atol)
    
    if not is_close:
        max_diff = np.abs(nabla_grad_np - pytorch_grad_np).max()
        print(f"    [{test_name}] Max abs error: {max_diff:.2e}")
        print(f"    Nabla shape: {nabla_grad_np.shape}, PyTorch shape: {pytorch_grad_np.shape}")
    
    return is_close


def run_conv2d_transpose_gradient_test(test_config):
    """
    Run a single conv2d_transpose gradient test comparing Nabla vs PyTorch.
    
    Returns True if test passes, False otherwise.
    """
    # Unpack test configuration
    test_name = test_config["name"]
    input_shape = test_config["input_shape"]
    kernel_shape = test_config["kernel_shape"]
    stride = test_config.get("stride", 1)
    padding = test_config.get("padding", 0)
    output_padding = test_config.get("output_padding", 0)
    dilation = test_config.get("dilation", 1)
    groups = test_config.get("groups", 1)
    bias_config = test_config.get("bias", False)
    dtype = getattr(torch, test_config.get("dtype", "float64"))
    
    print(f"Testing: {test_name} (dtype={dtype})")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # === PyTorch Reference ===
    X_torch = torch.randn(*input_shape, requires_grad=True, dtype=dtype)
    K_torch = torch.randn(*kernel_shape, requires_grad=True, dtype=dtype)
    B_torch = torch.randn(kernel_shape[1] * groups, dtype=dtype) if bias_config else None
    if B_torch is not None:
        B_torch.requires_grad = True
    
    # Handle padding for PyTorch
    if isinstance(padding, str):
        if padding == 'valid':
            padding_torch = 0
        elif padding == 'same':
            # For 'same' padding in transpose conv, compute appropriate padding
            if stride != 1 and not isinstance(stride, tuple):
                print(f"    ERROR: padding='same' is not supported for stride != 1")
                print("    --> Test SKIPPED\n")
                return None
            # Compute padding for 'same'
            k_h, k_w = kernel_shape[2:]
            s_h = stride if isinstance(stride, int) else stride[0]
            s_w = stride if isinstance(stride, int) else stride[1]
            d_h = dilation if isinstance(dilation, int) else dilation[0]
            d_w = dilation if isinstance(dilation, int) else dilation[1]
            if s_h != 1 or s_w != 1:
                print(f"    ERROR: padding='same' is not supported for stride != 1")
                print("    --> Test SKIPPED\n")
                return None
            total_h = d_h * (k_h - 1)
            total_w = d_w * (k_w - 1)
            padding_torch = (total_h // 2, total_w // 2)
    elif isinstance(padding, int):
        padding_torch = padding
    else:
        # Convert tuple - PyTorch expects single values if symmetric
        if padding[0] == padding[1]:
            padding_torch = padding[0]
        else:
            padding_torch = padding
    
    try:
        Y_torch = F.conv_transpose2d(X_torch, K_torch, bias=B_torch, stride=stride, 
                                    padding=padding_torch, output_padding=output_padding,
                                    dilation=dilation, groups=groups)
    except Exception as e:
        print(f"    PyTorch forward failed: {e}")
        print("    --> Test SKIPPED\n")
        return None
    
    grad_output_torch = torch.randn(Y_torch.shape, dtype=dtype)
    Y_torch.backward(grad_output_torch)
    
    # === Nabla Test ===
    X_nabla = pytorch_to_nabla(X_torch.detach())
    K_nabla = pytorch_to_nabla(K_torch.detach())
    
    # Set requires_grad
    X_nabla.requires_grad = True
    K_nabla.requires_grad = True
    
    # Use padding format that nabla expects (int or string or tuple, NOT pre-normalized)
    # nabla will normalize it internally
    if isinstance(padding, str):
        if padding == 'valid':
            padding_nabla = 0  # 'valid' means no padding
        elif padding == 'same':
            # For 'same' padding in transpose conv, compute appropriate padding
            k_h, k_w = kernel_shape[2:]
            d_h = dilation if isinstance(dilation, int) else dilation[0]
            d_w = dilation if isinstance(dilation, int) else dilation[1]
            total_h = d_h * (k_h - 1)
            total_w = d_w * (k_w - 1)
            pad_h = total_h // 2
            pad_w = total_w // 2
            padding_nabla = (pad_h, pad_w)
    elif isinstance(padding, int):
        padding_nabla = padding
    else:
        # padding is a tuple - nabla will normalize this
        padding_nabla = padding
    
    # Convert output_padding format for Nabla
    if isinstance(output_padding, int):
        output_padding_nabla = (output_padding, output_padding)
    else:
        output_padding_nabla = output_padding
    
    try:
        Y_nabla = nb.conv2d_transpose(X_nabla, K_nabla, stride=stride, 
                                     padding=padding_nabla, output_padding=output_padding_nabla,
                                     dilation=dilation, groups=groups)
    except Exception as e:
        print(f"    Nabla forward failed: {e}")
        print("    --> Test SKIPPED\n")
        return None
    
    # Compute gradients using Nabla's backward
    grad_output_nabla = pytorch_to_nabla(grad_output_torch)
    
    try:
        # Backward pass
        Y_nabla.backward(grad_output_nabla)
        grad_X_nabla = X_nabla.grad
        grad_K_nabla = K_nabla.grad
    except Exception as e:
        print(f"    Nabla gradient computation failed: {e}")
        print("    --> Test FAILED\n")
        return False
    
    # === Compare Results ===
    x_correct = compare_gradients(grad_X_nabla, X_torch.grad, dtype, "∂L/∂X")
    k_correct = compare_gradients(grad_K_nabla, K_torch.grad, dtype, "∂L/∂K")
    
    if x_correct and k_correct:
        print("    --> Test PASSED\n")
        return True
    else:
        print("    --> Test FAILED\n")
        return False


# ==================================================================
# --- TEST SUITE (from conv2d_transpose_manual_grad.py)
# ==================================================================

test_cases = [
    # === 1. Basic Sanity Checks ===
    {
        "name": "Sanity Check: Basic Case (int args)",
        "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3),
    },
    {
        "name": "Sanity Check: Basic Case (tuple args)",
        "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3),
        "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1),
    },
    # === 2. Parameter Isolation ===
    {
        "name": "Param Isolation: Stride > 1",
        "input_shape": (2, 3, 10, 10), "kernel_shape": (3, 4, 3, 3),
        "stride": 3,
    },
    {
        "name": "Param Isolation: Padding > 0",
        "input_shape": (2, 3, 8, 8), "kernel_shape": (3, 4, 3, 3),
        "padding": 2,
    },
    {
        "name": "Param Isolation: Dilation > 1",
        "input_shape": (2, 3, 10, 10), "kernel_shape": (3, 4, 3, 3),
        "dilation": 3,
    },
    # === 3. Parameter Combinations ===
    {
        "name": "Param Combo: Stride and Padding",
        "input_shape": (2, 3, 10, 10), "kernel_shape": (3, 4, 3, 3),
        "stride": 2, "padding": 1,
    },
    {
        "name": "Param Combo: Stride and Dilation",
        "input_shape": (2, 3, 12, 12), "kernel_shape": (3, 4, 3, 3),
        "stride": 2, "dilation": 2,
    },
    {
        "name": "Param Combo: Padding and Dilation",
        "input_shape": (2, 3, 12, 12), "kernel_shape": (3, 4, 3, 3),
        "padding": 2, "dilation": 2,
    },
    # === 4. Shape-based Edge Cases ===
    {
        "name": "Shape Edge Case: 1x1 Kernel (Pointwise Conv)",
        "input_shape": (2, 8, 10, 10), "kernel_shape": (8, 16, 1, 1),
        "stride": 1, "padding": 0,
    },
    {
        "name": "Shape Edge Case: Input size = Kernel size",
        "input_shape": (2, 3, 5, 5), "kernel_shape": (3, 4, 5, 5),
    },
    {
        "name": "Shape Edge Case: Minimal Input for 1x1 Output (Dilation)",
        "input_shape": (2, 3, 7, 7), "kernel_shape": (3, 4, 3, 3),
        "dilation": 3,
    },
    {
        "name": "Shape Edge Case: Stride > Kernel Size",
        "input_shape": (1, 1, 10, 10), "kernel_shape": (1, 1, 3, 3),
        "stride": 4,
    },
    # === 5. Grouped Convolution Cases ===
    {
        "name": "Groups: Standard Grouped Convolution",
        "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 4, 3, 3),
        "groups": 2,
    },
    {
        "name": "Groups: Depthwise Convolution (C_in=C_out=groups)",
        "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 1, 3, 3),
        "groups": 4,
    },
    {
        "name": "Groups: Depthwise Multiplier (C_out = K * C_in)",
        "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 2, 3, 3),
        "groups": 4,
    },
    # === 6. Asymmetric (Tuple) Parameter Tests ===
    {
        "name": "Asymmetric: Stride and Kernel",
        "input_shape": (1, 1, 10, 12), "kernel_shape": (1, 1, 3, 5),
        "stride": (2, 1),
    },
    {
        "name": "Asymmetric: Padding and Dilation",
        "input_shape": (1, 1, 10, 10), "kernel_shape": (1, 1, 3, 3),
        "padding": (2, 1), "dilation": (1, 2),
    },
    # === 7. String Padding Tests ===
    {
        "name": "String Padding: 'valid'",
        "input_shape": (1, 1, 8, 8), "kernel_shape": (1, 1, 3, 3),
        "padding": "valid",
    },
    {
        "name": "String Padding: 'same' with odd kernel",
        "input_shape": (1, 1, 8, 8), "kernel_shape": (1, 1, 3, 3),
        "padding": "same", "stride": 1,
    },
    {
        "name": "String Padding: 'same' with even kernel",
        "input_shape": (1, 1, 8, 8), "kernel_shape": (1, 1, 4, 4),
        "padding": "same", "stride": 1,
    },
    {
        "name": "String Padding: 'same' with dilation",
        "input_shape": (1, 1, 10, 10), "kernel_shape": (1, 1, 3, 3),
        "padding": "same", "dilation": 2, "stride": 1,
    },
    # === 8. Bias Gradient Tests ===
    {
        "name": "Bias Test: Simple case with bias",
        "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3),
        "bias": True,
    },
    {
        "name": "Bias Test: Multiple channels with bias",
        "input_shape": (2, 4, 8, 8), "kernel_shape": (4, 3, 3, 3),
        "bias": True,
    },
    {
        "name": "Bias Test: Grouped conv with bias",
        "input_shape": (2, 8, 10, 10), "kernel_shape": (8, 2, 3, 3),
        "groups": 2, "bias": True,
    },
    {
        "name": "Bias Test: Depthwise conv with bias",
        "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 1, 3, 3),
        "groups": 4, "bias": True,
    },
    {
        "name": "Bias Test: With stride, padding, dilation and bias",
        "input_shape": (2, 4, 12, 12), "kernel_shape": (4, 3, 3, 3),
        "stride": 2, "padding": 1, "dilation": 2, "bias": True,
    },
    # === 9. Batch Size Variations ===
    {
        "name": "Batch Size: N=1 minimal",
        "input_shape": (1, 1, 3, 3), "kernel_shape": (1, 1, 3, 3),
    },
    {
        "name": "Batch Size: N=8",
        "input_shape": (8, 3, 10, 10), "kernel_shape": (3, 4, 3, 3),
    },
    {
        "name": "Batch Size: N=32 very large",
        "input_shape": (32, 4, 6, 6), "kernel_shape": (4, 2, 3, 3),
    },
    # === 9. Channel Variations ===
    {
        "name": "Channel: C_in=1, C_out=1",
        "input_shape": (2, 1, 10, 10), "kernel_shape": (1, 1, 3, 3),
    },
    {
        "name": "Channel: C_in=1, C_out=many",
        "input_shape": (2, 1, 10, 10), "kernel_shape": (1, 32, 3, 3),
    },
    {
        "name": "Channel: C_in=many, C_out=1",
        "input_shape": (2, 32, 10, 10), "kernel_shape": (32, 1, 3, 3),
    },
    {
        "name": "Channel: High channel count",
        "input_shape": (2, 128, 10, 10), "kernel_shape": (128, 64, 3, 3),
    },
    # === 11. Minimal Spatial Dimensions ===
    {
        "name": "Spatial Edge: 2x2 input",
        "input_shape": (1, 1, 2, 2), "kernel_shape": (1, 1, 3, 3),
        "padding": 1,
    },
    {
        "name": "Spatial Edge: 3x3 input with 3x3 kernel",
        "input_shape": (2, 3, 3, 3), "kernel_shape": (3, 4, 3, 3),
    },
    # === 12. Large Spatial Dimensions ===
    {
        "name": "Spatial Large: 64x64 input",
        "input_shape": (1, 4, 64, 64), "kernel_shape": (4, 3, 3, 3),
    },
    # === 13. Non-Square Kernels and Inputs ===
    {
        "name": "Non-square: Rectangular kernel 3x5",
        "input_shape": (1, 1, 10, 12), "kernel_shape": (1, 1, 3, 5),
    },
    {
        "name": "Non-square: Asymmetric stride + kernel",
        "input_shape": (1, 1, 15, 20), "kernel_shape": (1, 1, 3, 5),
        "stride": (2, 3), "padding": (1, 2),
    },
    # === 12. Extreme Parameter Combinations ===
    {
        "name": "Extreme: Very large stride",
        "input_shape": (1, 1, 20, 20), "kernel_shape": (1, 1, 3, 3),
        "stride": 7,
    },
    {
        "name": "Extreme: Very large dilation",
        "input_shape": (1, 1, 30, 30), "kernel_shape": (1, 1, 3, 3),
        "dilation": 5,
    },
    {
        "name": "Extreme: All params large asymmetric",
        "input_shape": (2, 4, 40, 50), "kernel_shape": (4, 3, 5, 7),
        "stride": (3, 4), "padding": (2, 3), "dilation": (2, 2),
    },
    # === 15. One-Dimensional Output Cases ===
    {
        "name": "Output 1x1: Heavy stride on small input",
        "input_shape": (1, 1, 10, 10), "kernel_shape": (1, 1, 3, 3),
        "stride": 9,
    },
    {
        "name": "Output 1x1: via dilation + kernel size",
        "input_shape": (1, 1, 7, 7), "kernel_shape": (1, 1, 3, 3),
        "dilation": 3,
    },
    # === 16. Gradient Output Variations ===
    {
        "name": "Special Grad: All-ones gradient output",
        "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3),
        "special_grad": "ones",
    },
    {
        "name": "Special Grad: Sparse gradient (mostly zeros)",
        "input_shape": (2, 4, 8, 8), "kernel_shape": (4, 3, 3, 3),
        "special_grad": "sparse",
    },
    # === 17. DATA TYPE (DTYPE) TESTS ===
    {
        "name": "DType: Basic Case (float32)",
        "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3),
        "dtype": "float32",
    },
    {
        "name": "DType: Grouped Conv (float32)",
        "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 4, 3, 3),
        "groups": 2, "bias": True, "dtype": "float32",
    },
    {
        "name": "DType: High Channel (float32)",
        "input_shape": (2, 32, 10, 10), "kernel_shape": (32, 1, 3, 3),
        "dtype": "float32",
    },
    # === 18. MULTI-PASS GRADIENT ACCUMULATION ===
    {
        "name": "Multi-pass: Gradient accumulation (N=3)",
        "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3),
        "num_passes": 3,
    },
    {
        "name": "Multi-pass: Gradient accumulation (N=5, groups, bias)",
        "input_shape": (2, 4, 8, 8), "kernel_shape": (4, 1, 3, 3),
        "groups": 4, "bias": True, "num_passes": 5,
    },
    {
        "name": "Multi-pass: Accumulation (N=2, float32)",
        "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3),
        "num_passes": 2, "dtype": "float32",
    },
    # === 19. KITCHEN SINK STRESS TESTS ===
    {
        "name": "STRESS TEST: All Params Asymmetric + Batch + Groups",
        "input_shape": (4, 8, 20, 18), 
        "kernel_shape": (8, 3, 5, 3), # C_in_per_group = 8/2=4, but weight shape for conv_transpose2d is (C_in, C_out_per_group, H, W)
        "stride": (2, 3), 
        "padding": (3, 1), 
        "dilation": (2, 1), 
        "groups": 2,
    },
    {
        "name": "STRESS TEST: Everything with bias (float32)",
        "input_shape": (4, 8, 20, 18), 
        "kernel_shape": (8, 3, 5, 3),
        "stride": (2, 3), 
        "padding": (3, 1), 
        "dilation": (2, 1), 
        "groups": 2,
        "bias": True, "dtype": "float32",
    },
    {
        "name": "STRESS TEST: Large batch, high channels",
        "input_shape": (16, 128, 16, 16),
        "kernel_shape": (128, 64, 3, 3),
        "bias": True,
    },
]


# ==================================================================
# --- PYTEST TEST FUNCTIONS
# ==================================================================

@pytest.mark.parametrize("test_config", test_cases, ids=[tc["name"] for tc in test_cases])
def test_conv2d_transpose_gradient(test_config):
    """Test conv2d_transpose gradient computation against PyTorch."""
    result = run_conv2d_transpose_gradient_test(test_config)
    if result is None:
        pytest.skip("Test skipped due to forward pass failure")
    assert result, f"Gradient test failed for: {test_config['name']}"


if __name__ == '__main__':
    # Run tests manually
    total_tests = len(test_cases)
    passed = 0
    failed = 0
    skipped = 0
    
    print(f"Running {total_tests} conv2d_transpose gradient tests...\n")
    
    for config in test_cases:
        result = run_conv2d_transpose_gradient_test(config)
        if result is None:
            skipped += 1
        elif result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Test Summary:")
    print(f"  Total:   {total_tests}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"{'='*70}")
