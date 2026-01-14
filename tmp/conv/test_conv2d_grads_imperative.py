#!/usr/bin/env python
"""
Gradient correctness tests for nb.conv2d using imperative .backward(),
comparing against PyTorch autograd on the same inputs/weights/bias and dY.

This recreates a comprehensive set of cases similar in spirit to the
manual-gradient suites, but now validating Nabla's VJP against PyTorch.
"""

import numpy as np
import nabla as nb

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    print("PyTorch not available - skipping tests:", e)


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def _make_padding_for_same_2d(kernel_shape, dilation):
    # Only valid for stride == 1
    if isinstance(dilation, int):
        d_h = d_w = dilation
    else:
        d_h, d_w = dilation
    k_h, k_w = kernel_shape
    total_h = d_h * (k_h - 1)
    total_w = d_w * (k_w - 1)
    return (total_h // 2, total_w // 2)


def _gen_grad_output_like(y_pt, special_grad: str | None, dtype):
    go = torch.randn_like(y_pt)
    if special_grad == "ones":
        go.fill_(1.0)
    elif special_grad == "sparse":
        go *= (torch.rand_like(go) > 0.1)
    elif special_grad == "small":
        go *= 1e-5
    elif special_grad == "large":
        go *= 1e3
    return go


def _compare_arrays(a: np.ndarray, b: np.ndarray, dtype) -> tuple[bool, float]:
    if dtype == np.float64:
        rtol, atol = 1e-5, 1e-8
    else:
        rtol, atol = 1e-3, 1e-5
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    max_err = float(np.max(np.abs(a - b))) if not ok else 0.0
    return ok, max_err


def verify_conv2d_grads_against_torch(test_config: dict) -> bool:
    name = test_config["name"]
    input_shape = test_config["input_shape"]  # (N, C_in, H, W)
    weight_shape = test_config["kernel_shape"]  # (C_out, C_in/groups, K_h, K_w)
    stride = test_config.get("stride", 1)
    padding = test_config.get("padding", 0)
    dilation = test_config.get("dilation", 1)
    groups = test_config.get("groups", 1)
    use_bias = test_config.get("bias", False)
    special_grad = test_config.get("special_grad", None)
    num_passes = int(test_config.get("num_passes", 1))
    dtype_name = test_config.get("dtype", "float32")

    # dtype
    np_dtype = np.float64 if dtype_name == "float64" else np.float32
    torch_dtype = torch.float64 if dtype_name == "float64" else torch.float32

    print(f"--- Conv2D Grads Test: {name} (dtype={dtype_name}) ---")
    _set_seed(42)

    # Prepare tensors
    N, C_in, H_in, W_in = input_shape
    C_out, C_in_div_g, K_h, K_w = weight_shape

    # Convert string paddings
    pad_arg = padding
    if isinstance(padding, str):
        if padding == "valid":
            pad_arg = 0
        elif padding == "same":
            # Only support for stride == 1
            s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
            if s_h != 1 or s_w != 1:
                print("    SKIP: padding='same' with stride != 1 not supported in this tester")
                return True  # Skip instead of failing the whole suite
            ph, pw = _make_padding_for_same_2d((K_h, K_w), dilation)
            pad_arg = (ph, pw)
        else:
            print(f"    SKIP: unknown padding string '{padding}'")
            return True

    # Random data
    x_np = np.random.randn(*input_shape).astype(np_dtype)
    w_np = np.random.randn(*weight_shape).astype(np_dtype)
    b_np = np.random.randn(C_out).astype(np_dtype) if use_bias else None

    # PyTorch reference
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch_dtype)
    # Nabla conv2d weight gradients for groups>1 are not implemented yet
    w_requires = (groups == 1)
    w_pt = torch.tensor(w_np, requires_grad=w_requires, dtype=torch_dtype)
    b_pt = torch.tensor(b_np, requires_grad=True, dtype=torch_dtype) if use_bias else None

    try:
        y_pt = F.conv2d(x_pt, w_pt, bias=b_pt, stride=stride, padding=pad_arg, dilation=dilation, groups=groups)
    except Exception as e:
        print(f"    ERROR torch forward: {e}")
        return False

    # Grad output
    go_pt = _gen_grad_output_like(y_pt, special_grad, torch_dtype)

    # Zero grads then backward num_passes
    for p in (x_pt, w_pt) + ((b_pt,) if b_pt is not None else tuple()):
        if p is not None and p.grad is not None:
            p.grad = None
    for i in range(num_passes):
        y_pt.backward(go_pt, retain_graph=(i < num_passes - 1))

    # Nabla computation (imperative)
    x_nb = nb.tensor(x_np).requires_grad_()
    w_nb = nb.tensor(w_np).requires_grad_(w_requires)
    b_nb = nb.tensor(b_np).requires_grad_() if use_bias else None

    try:
        y_nb = nb.conv2d(x_nb, w_nb, stride=stride, padding=padding, dilation=dilation, groups=groups)
        if use_bias:
            y_nb = y_nb + b_nb.reshape((1, C_out, 1, 1))
    except Exception as e:
        print(f"    ERROR nabla forward: {e}")
        return False

    go_nb = nb.tensor(go_pt.detach().cpu().numpy()).astype(y_nb.dtype)

    # Clear old grads
    x_nb.grad = None
    w_nb.grad = None
    if b_nb is not None:
        b_nb.grad = None

    # Backward accumulation
    for i in range(num_passes):
        y_nb.backward(go_nb, retain_graph=(i < num_passes - 1))

    # Compare grads
    ok_x, err_x = _compare_arrays(x_nb.grad.to_numpy(), x_pt.grad.detach().cpu().numpy(), np_dtype)
    if w_requires:
        ok_w, err_w = _compare_arrays(w_nb.grad.to_numpy(), w_pt.grad.detach().cpu().numpy(), np_dtype)
    else:
        ok_w, err_w = True, 0.0
    if use_bias:
        ok_b, err_b = _compare_arrays(b_nb.grad.to_numpy(), b_pt.grad.detach().cpu().numpy(), np_dtype)
    else:
        ok_b, err_b = True, 0.0

    print(f"    dX match: {ok_x}  max|diff|={err_x:.3e}")
    if w_requires:
        print(f"    dW match: {ok_w}  max|diff|={err_w:.3e}")
    else:
        print("    dW skipped (groups>1 not supported for conv2d weight VJP in Nabla)")
    if use_bias:
        print(f"    dB match: {ok_b}  max|diff|={err_b:.3e}")

    return ok_x and ok_w and ok_b


# Test matrix (2D conv) - matching conv2d_manual_grad.py test suite
conv2d_test_cases = [
    # === 1. Basic Sanity Checks ===
    {"name": "Sanity: Basic (int)", "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3)},
    {"name": "Sanity: Basic (tuple)", "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1)},
    # === 2. Parameter Isolation ===
    {"name": "Param: Stride>1", "input_shape": (2, 3, 10, 10), "kernel_shape": (4, 3, 3, 3), "stride": 3},
    {"name": "Param: Padding>0", "input_shape": (2, 3, 8, 8), "kernel_shape": (4, 3, 3, 3), "padding": 2},
    {"name": "Param: Dilation>1", "input_shape": (2, 3, 10, 10), "kernel_shape": (4, 3, 3, 3), "dilation": 3},
    # === 3. Parameter Combinations ===
    {"name": "Combo: Stride+Padding", "input_shape": (2, 3, 10, 10), "kernel_shape": (4, 3, 3, 3), "stride": 2, "padding": 1},
    {"name": "Combo: Stride+Dilation", "input_shape": (2, 3, 12, 12), "kernel_shape": (4, 3, 3, 3), "stride": 2, "dilation": 2},
    {"name": "Combo: Padding+Dilation", "input_shape": (2, 3, 12, 12), "kernel_shape": (4, 3, 3, 3), "padding": 2, "dilation": 2},
    # === 4. Shape-based Edge Cases ===
    {"name": "Shape: 1x1 Kernel", "input_shape": (2, 8, 10, 10), "kernel_shape": (16, 8, 1, 1)},
    {"name": "Shape: Input==Kernel", "input_shape": (2, 3, 5, 5), "kernel_shape": (4, 3, 5, 5)},
    {"name": "Shape: Min Input for 1x1 Out", "input_shape": (2, 3, 7, 7), "kernel_shape": (4, 3, 3, 3), "dilation": 3},
    {"name": "Shape: Stride>Kernel", "input_shape": (1, 1, 10, 10), "kernel_shape": (1, 1, 3, 3), "stride": 4},
    # === 5. Grouped Convolution Cases ===
    {"name": "Groups: Standard", "input_shape": (2, 4, 10, 10), "kernel_shape": (8, 2, 3, 3), "groups": 2},
    {"name": "Groups: Depthwise", "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 1, 3, 3), "groups": 4},
    {"name": "Groups: Depthwise Mult", "input_shape": (2, 4, 10, 10), "kernel_shape": (8, 1, 3, 3), "groups": 4},
    # === 6. Asymmetric (Tuple) Parameter Tests ===
    {"name": "Asym: Stride+Kernel", "input_shape": (1, 1, 10, 12), "kernel_shape": (1, 1, 3, 5), "stride": (2, 1)},
    {"name": "Asym: Padding+Dilation", "input_shape": (1, 1, 10, 10), "kernel_shape": (1, 1, 3, 3), "padding": (2, 1), "dilation": (1, 2)},
    # === 7. String Padding Tests ===
    {"name": "Padding: 'valid'", "input_shape": (1, 1, 8, 8), "kernel_shape": (1, 1, 3, 3), "padding": "valid"},
    {"name": "Padding: 'same' odd", "input_shape": (1, 1, 8, 8), "kernel_shape": (1, 1, 3, 3), "padding": "same", "stride": 1},
    {"name": "Padding: 'same' even", "input_shape": (1, 1, 8, 8), "kernel_shape": (1, 1, 4, 4), "padding": "same", "stride": 1},
    {"name": "Padding: 'same' dilated", "input_shape": (1, 1, 10, 10), "kernel_shape": (1, 1, 3, 3), "padding": "same", "dilation": 2, "stride": 1},
    # === 8. Bias Gradient Tests ===
    {"name": "Bias: Simple", "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3), "bias": True},
    {"name": "Bias: Multi-channel", "input_shape": (2, 3, 8, 8), "kernel_shape": (4, 3, 3, 3), "bias": True},
    {"name": "Bias: Grouped", "input_shape": (2, 4, 10, 10), "kernel_shape": (8, 2, 3, 3), "groups": 2, "bias": True},
    {"name": "Bias: Depthwise", "input_shape": (2, 4, 10, 10), "kernel_shape": (4, 1, 3, 3), "groups": 4, "bias": True},
    {"name": "Bias: All Params", "input_shape": (2, 3, 12, 12), "kernel_shape": (4, 3, 3, 3), "stride": 2, "padding": 1, "dilation": 2, "bias": True},
    # === 9. Batch Size Variations ===
    {"name": "Batch: N=1 minimal", "input_shape": (1, 1, 3, 3), "kernel_shape": (1, 1, 3, 3)},
    {"name": "Batch: N=8", "input_shape": (8, 3, 10, 10), "kernel_shape": (4, 3, 3, 3)},
    {"name": "Batch: N=32 large", "input_shape": (32, 2, 6, 6), "kernel_shape": (4, 2, 3, 3)},
    # === 10. Channel Variations ===
    {"name": "Channel: C_in=1, C_out=1", "input_shape": (2, 1, 10, 10), "kernel_shape": (1, 1, 3, 3)},
    {"name": "Channel: C_in=1, C_out=many", "input_shape": (2, 1, 10, 10), "kernel_shape": (32, 1, 3, 3)},
    {"name": "Channel: C_in=many, C_out=1", "input_shape": (2, 32, 10, 10), "kernel_shape": (1, 32, 3, 3)},
    {"name": "Channel: High count", "input_shape": (2, 64, 10, 10), "kernel_shape": (128, 64, 3, 3)},
    # === 11. Minimal Spatial Dimensions ===
    {"name": "Spatial: 2x2 input", "input_shape": (1, 1, 2, 2), "kernel_shape": (1, 1, 3, 3), "padding": 1},
    {"name": "Spatial: 3x3 input, 3x3 kernel", "input_shape": (2, 3, 3, 3), "kernel_shape": (4, 3, 3, 3)},
    # === 12. Large Spatial Dimensions ===
    {"name": "Spatial: 64x64 input", "input_shape": (1, 3, 64, 64), "kernel_shape": (4, 3, 3, 3)},
    # === 13. Non-Square Kernels and Inputs ===
    {"name": "Non-square: 3x5 kernel", "input_shape": (1, 1, 10, 12), "kernel_shape": (1, 1, 3, 5)},
    {"name": "Non-square: Asym stride+kernel", "input_shape": (1, 1, 15, 20), "kernel_shape": (1, 1, 3, 5), "stride": (2, 3), "padding": (1, 2)},
    # === 14. Extreme Parameter Combinations ===
    {"name": "Extreme: Very large stride", "input_shape": (1, 1, 20, 20), "kernel_shape": (1, 1, 3, 3), "stride": 7},
    {"name": "Extreme: Very large dilation", "input_shape": (1, 1, 30, 30), "kernel_shape": (1, 1, 3, 3), "dilation": 5},
    {"name": "Extreme: All params large asym", "input_shape": (2, 3, 40, 50), "kernel_shape": (4, 3, 5, 7), "stride": (3, 4), "padding": (2, 3), "dilation": (2, 2)},
    # === 15. One-Dimensional Output Cases ===
    {"name": "Output 1x1: Heavy stride", "input_shape": (1, 1, 10, 10), "kernel_shape": (1, 1, 3, 3), "stride": 9},
    {"name": "Output 1x1: Via dilation", "input_shape": (1, 1, 7, 7), "kernel_shape": (1, 1, 3, 3), "dilation": 3},
    # === 16. Gradient Output Variations ===
    {"name": "Special Grad: All-ones", "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3), "special_grad": "ones"},
    {"name": "Special Grad: Sparse", "input_shape": (2, 3, 8, 8), "kernel_shape": (4, 3, 3, 3), "special_grad": "sparse"},
    # === 17. Data Type Tests ===
    {"name": "DType: float32 basic", "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3), "dtype": "float32"},
    {"name": "DType: float32 grouped+bias", "input_shape": (2, 4, 10, 10), "kernel_shape": (8, 2, 3, 3), "groups": 2, "bias": True, "dtype": "float32"},
    {"name": "DType: float32 high channel", "input_shape": (2, 32, 10, 10), "kernel_shape": (1, 32, 3, 3), "dtype": "float32"},
    # === 18. Multi-Pass Gradient Accumulation ===
    {"name": "Multi-pass: N=3", "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3), "num_passes": 3},
    {"name": "Multi-pass: N=5 groups+bias", "input_shape": (2, 4, 8, 8), "kernel_shape": (4, 1, 3, 3), "groups": 4, "bias": True, "num_passes": 5},
    {"name": "Multi-pass: N=2 float32", "input_shape": (1, 1, 5, 5), "kernel_shape": (1, 1, 3, 3), "num_passes": 2, "dtype": "float32"},
    # === 19. Kitchen Sink Stress Tests ===
    {"name": "STRESS: All params asym+groups", "input_shape": (4, 6, 20, 18), "kernel_shape": (8, 3, 5, 3), "stride": (2, 3), "padding": (3, 1), "dilation": (2, 1), "groups": 2},
    {"name": "STRESS: Everything+bias float32", "input_shape": (4, 6, 20, 18), "kernel_shape": (8, 3, 5, 3), "stride": (2, 3), "padding": (3, 1), "dilation": (2, 1), "groups": 2, "bias": True, "dtype": "float32"},
    {"name": "STRESS: Large batch+channels", "input_shape": (16, 64, 16, 16), "kernel_shape": (128, 64, 3, 3), "bias": True},
]


def run_all_conv2d_tests():
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping all conv2d grad tests.")
        return True

    all_ok = True
    for cfg in conv2d_test_cases:
        ok = verify_conv2d_grads_against_torch(cfg)
        all_ok = all_ok and ok
    print("\nConv2D grad tests:", "PASSED" if all_ok else "FAILED")
    return all_ok


if __name__ == "__main__":
    run_all_conv2d_tests()
