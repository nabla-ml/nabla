#!/usr/bin/env python
import torch
import torch.nn.functional as F

def verify_conv_transpose2d_gradients(test_config: dict):
    """
    A testing function to verify the manually derived gradients for F.conv_transpose2d
    against PyTorch's autograd gradients for a given configuration.
    
    This version is adapted from the conv2d version, supporting similar features.
    """
    # === 1. Unpack Test Configuration ===
    test_name = test_config["name"]
    input_shape = test_config["input_shape"]
    kernel_shape = test_config["kernel_shape"]
    stride = test_config.get("stride", 1)
    padding = test_config.get("padding", 0)
    output_padding = test_config.get("output_padding", 0)
    dilation = test_config.get("dilation", 1)
    groups = test_config.get("groups", 1)
    bias_config = test_config.get("bias", False)
    special_grad = test_config.get("special_grad", None)
    num_passes = test_config.get("num_passes", 1)
    
    dtype = getattr(torch, test_config.get("dtype", "float64"))

    print(f"--- Running Test Case: {test_name} (dtype={dtype}) ---")
    
    torch.manual_seed(42)

    # === 2. Parse Parameters and Compute Padding ===
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    d_h, d_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    op_h, op_w = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding

    if isinstance(padding, str):
        if padding == 'valid':
            pad_h0 = pad_h1 = pad_w0 = pad_w1 = 0
        elif padding == 'same':
            if s_h != 1 or s_w != 1:
                print(f"    ERROR: padding='same' is not supported for stride != 1")
                print("    --> Test SKIPPED.\n")
                return
            total_h = d_h * (kernel_shape[2] - 1)
            pad_h0 = total_h // 2; pad_h1 = total_h - pad_h0
            total_w = d_w * (kernel_shape[3] - 1)
            pad_w0 = total_w // 2; pad_w1 = total_w - pad_w0
            op_h = op_w = 0
        else:
            print(f"    ERROR: Unknown string padding '{padding}'")
            print("    --> Test SKIPPED.\n")
            return
    else:
        p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
        pad_h0 = pad_h1 = p_h; pad_w0 = pad_w1 = p_w

    # === 3. Setup Tensors ===
    (N, C_in, H_in, W_in) = input_shape
    (C_in_kernel, C_out_g, K_h, K_w) = kernel_shape
    C_out = C_out_g * groups
    
    X = torch.randn(*input_shape, requires_grad=True, dtype=dtype)
    K = torch.randn(*kernel_shape, requires_grad=True, dtype=dtype)
    B = torch.randn(C_out, dtype=dtype) if bias_config else None
    if B is not None: B.requires_grad = True

    # === 4. Forward Pass ===
    try:
        Y = F.conv_transpose2d(X, K, bias=B, stride=(s_h, s_w), padding=(pad_h0, pad_w0), output_padding=(op_h, op_w), groups=groups, dilation=(d_h, d_w))
    except Exception as e:
        print(f"    ERROR during forward pass: {e}")
        print("    --> Test SKIPPED.\n")
        return
    (H_out, W_out) = Y.shape[2:]

    # === 5. Autograd Backward Pass (with Accumulation) ===
    grad_output = torch.randn(Y.shape, dtype=dtype)
    if special_grad == "ones": grad_output.fill_(1.0)
    elif special_grad == "sparse": grad_output *= (torch.rand_like(grad_output) > 0.1)
    elif special_grad == "small": grad_output *= 1e-5
    elif special_grad == "large": grad_output *= 1e3
    
    if X.grad: X.grad.zero_()
    if K.grad: K.grad.zero_()
    if B is not None and B.grad: B.grad.zero_()

    for i in range(num_passes):
        Y.backward(grad_output, retain_graph=(i < num_passes - 1))

    # === 6. Manual Gradient Calculation ===
    # 1. Gradient with respect to the Input (∂L/∂X)
    s_prime_h = s_h
    s_prime_w = s_w
    d_prime_h = d_h
    d_prime_w = d_w

    H_prime = (H_in - 1) * s_prime_h + d_prime_h * (K_h - 1) + 1
    W_prime = (W_in - 1) * s_prime_w + d_prime_w * (K_w - 1) + 1

    total_padding_h = H_prime - H_out
    total_padding_w = W_prime - W_out

    if total_padding_h >= 0:
        pad_h0 = total_padding_h // 2
        pad_h1 = total_padding_h - pad_h0
        crop_h_start = 0
        crop_h_end_amount = 0
    else:
        pad_h0 = pad_h1 = 0
        crop_h_start = (-total_padding_h) // 2
        crop_h_end_amount = -total_padding_h - crop_h_start

    if total_padding_w >= 0:
        pad_w0 = total_padding_w // 2
        pad_w1 = total_padding_w - pad_w0
        crop_w_start = 0
        crop_w_end_amount = 0
    else:
        pad_w0 = pad_w1 = 0
        crop_w_start = (-total_padding_w) // 2
        crop_w_end_amount = -total_padding_w - crop_w_start

    # Apply crop if necessary
    h_end = -crop_h_end_amount if crop_h_end_amount > 0 else None
    w_end = -crop_w_end_amount if crop_w_end_amount > 0 else None
    grad_output_cropped = grad_output[:, :, crop_h_start:h_end, crop_w_start:w_end]

    # Apply padding
    grad_output_padded = F.pad(grad_output_cropped, (pad_w0, pad_w1, pad_h0, pad_h1), mode='constant', value=0)

    manual_grad_X = F.conv2d(grad_output_padded, K, stride=(s_prime_h, s_prime_w), padding=0, dilation=(d_prime_h, d_prime_w), groups=groups)

    # 2. Gradient with respect to the Kernel (∂L/∂K)
    k_grad_method = "unfold (general)"
    C_in_g = C_in // groups
    C_out_g = C_out // groups
    L = H_in * W_in
    grad_output_reshaped = grad_output.view(N * groups, C_out_g, H_out, W_out)
    unfolded_grad_output = F.unfold(grad_output_reshaped, (K_h, K_w), dilation=(d_h, d_w), padding=(pad_h0, pad_w0), stride=(s_h, s_w))
    unfolded_grad_output = unfolded_grad_output.view(N, groups, C_out_g * K_h * K_w, L).permute(1, 0, 3, 2).reshape(groups, N * L, C_out_g * K_h * K_w)
    x_reshaped = X.view(N, groups, C_in_g, L).permute(1, 2, 0, 3).reshape(groups, C_in_g, N * L)
    grad_K_bmm = x_reshaped @ unfolded_grad_output
    manual_grad_K = grad_K_bmm.view(groups, C_in_g, C_out_g, K_h, K_w).reshape(C_in, C_out_g, K_h, K_w)

    # 3. Gradient with respect to Bias (∂L/∂B)
    manual_grad_B = grad_output.sum(dim=(0, 2, 3)) if B is not None else None

    # 4. Scale Manual Gradients for Accumulation
    if num_passes > 1:
        manual_grad_X = manual_grad_X * num_passes
        manual_grad_K = manual_grad_K * num_passes
        if manual_grad_B is not None:
            manual_grad_B = manual_grad_B * num_passes

    # === 7. Verification ===
    if dtype == torch.float64:
        rtol, atol = 1e-5, 1e-8
    elif dtype == torch.float32:
        rtol, atol = 1e-3, 1e-5
    else: # Fallback for float16 or other low-precision
        rtol, atol = 1e-1, 1e-2

    x_correct = torch.allclose(X.grad, manual_grad_X, rtol=rtol, atol=atol)
    k_correct = torch.allclose(K.grad, manual_grad_K, rtol=rtol, atol=atol)
    b_correct = (B is None) or torch.allclose(B.grad, manual_grad_B, rtol=rtol, atol=atol)

    print(f"    ∂L/∂X correct: {x_correct}")
    print(f"    ∂L/∂K correct: {k_correct} (Method: {k_grad_method})")
    if B is not None: print(f"    ∂L/∂B correct: {b_correct}")
    else: print("    No bias.")
    
    if x_correct and k_correct and b_correct:
        print("    --> Test PASSED.\n")
    else:
        print("    --> Test FAILED.\n")
        if not x_correct:
            x_error = (X.grad - manual_grad_X).abs().max().item()
            print(f"    [X grad] Max abs error: {x_error:.2e}, Shapes: Autograd={X.grad.shape}, Manual={manual_grad_X.shape}")
        if not k_correct:
            k_error = (K.grad - manual_grad_K).abs().max().item()
            print(f"    [K grad] Max abs error: {k_error:.2e}, Shapes: Autograd={K.grad.shape}, Manual={manual_grad_K.shape}")
        if B is not None and not b_correct:
            b_error = (B.grad - manual_grad_B).abs().max().item()
            print(f"    [B grad] Max abs error: {b_error:.2e}, Shapes: Autograd={B.grad.shape}, Manual={manual_grad_B.shape}")
        print("")


# ==================================================================
# --- TEST SUITE
# ==================================================================
# Note: Test cases adapted from conv2d suite.
# Kernel shapes have been adjusted for conv_transpose2d convention:
# kernel_shape = (C_in, C_out // groups, K_h, K_w)
# Some tests may skip if forward pass fails due to size mismatches with stride >1; 
# output_padding can be added to configs if needed for specific tests.
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
    # === 5. Grouped Convolution Cases (Tests both k_grad methods) ===
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
        "groups": 4, # C_out (8) is 2 * C_in (4)
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
        "input_shape": (2, 3, 8, 8), "kernel_shape": (3, 4, 3, 3),
        "bias": True,
    },
    {
        "name": "Bias Test: Grouped conv with bias",
        "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 4, 3, 3),
        "groups": 2, "bias": True,
    },
    {
        "name": "Bias Test: Depthwise conv with bias",
        "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 1, 3, 3),
        "groups": 4, "bias": True,
    },
    {
        "name": "Bias Test: With stride, padding, dilation and bias",
        "input_shape": (2, 3, 12, 12), "kernel_shape": (3, 4, 3, 3),
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
        "input_shape": (32, 2, 6, 6), "kernel_shape": (2, 4, 3, 3),
    },
    # === 10. Channel Variations ===
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
        "input_shape": (2, 64, 10, 10), "kernel_shape": (64, 128, 3, 3),
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
        "input_shape": (1, 3, 64, 64), "kernel_shape": (3, 4, 3, 3),
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
    # === 14. Extreme Parameter Combinations ===
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
        "input_shape": (2, 3, 40, 50), "kernel_shape": (3, 4, 5, 7),
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
        "input_shape": (2, 3, 8, 8), "kernel_shape": (3, 4, 3, 3),
        "special_grad": "sparse",
    },
    # === 17. DATA TYPE (DTYPE) TESTS (NEW) ===
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
    # === 18. MULTI-PASS GRADIENT ACCUMULATION (NOW FUNCTIONAL) ===
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
        "input_shape": (4, 6, 20, 18), 
        "kernel_shape": (6, 4, 5, 3), 
        "stride": (2, 3), 
        "padding": (3, 1), 
        "dilation": (2, 1), 
        "groups": 2,
    },
    {
        "name": "STRESS TEST: Everything with bias (float32)",
        "input_shape": (4, 6, 20, 18), 
        "kernel_shape": (6, 4, 5, 3),
        "stride": (2, 3), 
        "padding": (3, 1), 
        "dilation": (2, 1), 
        "groups": 2,
        "bias": True, "dtype": "float32",
    },
    {
        "name": "STRESS TEST: Large batch, high channels",
        "input_shape": (16, 64, 16, 16),
        "kernel_shape": (64, 128, 3, 3),
        "bias": True,
    },
]

# --- Execute Test Suite ---
if __name__ == '__main__':
    total_tests = len(test_cases)
    print(f"Running {total_tests} test cases...\n")
    
    for config in test_cases:
        verify_conv_transpose2d_gradients(config)
    
    print(f"--- All {total_tests} test cases executed. ---")