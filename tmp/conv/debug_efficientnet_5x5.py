#!/usr/bin/env python
"""Debug the EfficientNet_5x5_early test case."""

import numpy as np
import nabla as nb
import torch
import torch.nn.functional as F

np.random.seed(42)

# Configuration from the failing test
batch, in_ch, out_ch, H, W, K, stride, padding = 16, 32, 64, 112, 112, 5, 1, 2

# Create inputs
input_np = np.random.randn(batch, in_ch, H, W).astype(np.float32) * 0.1
weight_np = np.random.randn(out_ch, in_ch, K, K).astype(np.float32) * 0.1

# Compute output shape
H_out = (H + 2 * padding - K) // stride + 1
W_out = (W + 2 * padding - K) // stride + 1
grad_output_np = np.random.randn(batch, out_ch, H_out, W_out).astype(np.float32) * 0.1

# PyTorch reference
input_pt = torch.from_numpy(input_np).requires_grad_(True)
weight_pt = torch.from_numpy(weight_np).requires_grad_(True)
grad_output_pt = torch.from_numpy(grad_output_np)

output_pt = F.conv2d(input_pt, weight_pt, stride=stride, padding=padding)
output_pt.backward(grad_output_pt)

# Nabla VJP
input_nb = nb.tensor(input_np)
weight_nb = nb.tensor(weight_np)
grad_output_nb = nb.tensor(grad_output_np)

def conv_fn(inp, weight):
    return nb.conv2d(inp, weight, stride=stride, padding=padding)

output_nb, vjp_fn = nb.vjp(conv_fn, input_nb, weight_nb)
grad_input_nb, grad_weight_nb = vjp_fn(grad_output_nb)

# Detailed comparison
print("=" * 60)
print("EfficientNet_5x5_early Debug")
print("=" * 60)
print(f"Config: batch={batch}, in_ch={in_ch}, out_ch={out_ch}, H={H}, W={W}")
print(f"        K={K}, stride={stride}, padding={padding}")
print()

# Check input gradients
grad_input_diff = np.abs(grad_input_nb.to_numpy() - input_pt.grad.numpy())
print(f"Input gradient:")
print(f"  Max abs diff: {grad_input_diff.max():.2e}")
print(f"  Mean abs diff: {grad_input_diff.mean():.2e}")
print(f"  Median abs diff: {np.median(grad_input_diff):.2e}")
print(f"  95th percentile: {np.percentile(grad_input_diff, 95):.2e}")
print(f"  99th percentile: {np.percentile(grad_input_diff, 99):.2e}")
input_match = np.allclose(grad_input_nb.to_numpy(), input_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
print(f"  Passes tolerance (rtol=1e-4, atol=1e-4): {input_match}")
print()

# Check weight gradients
grad_weight_diff = np.abs(grad_weight_nb.to_numpy() - weight_pt.grad.numpy())
print(f"Weight gradient:")
print(f"  Max abs diff: {grad_weight_diff.max():.2e}")
print(f"  Mean abs diff: {grad_weight_diff.mean():.2e}")
print(f"  Median abs diff: {np.median(grad_weight_diff):.2e}")
print(f"  95th percentile: {np.percentile(grad_weight_diff, 95):.2e}")
print(f"  99th percentile: {np.percentile(grad_weight_diff, 99):.2e}")
weight_match = np.allclose(grad_weight_nb.to_numpy(), weight_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
print(f"  Passes tolerance (rtol=1e-4, atol=1e-4): {weight_match}")
print()

# Try slightly relaxed tolerance
weight_match_relaxed = np.allclose(grad_weight_nb.to_numpy(), weight_pt.grad.numpy(), rtol=3e-4, atol=3e-4)
print(f"Weight gradient with relaxed tolerance (rtol=3e-4, atol=3e-4): {weight_match_relaxed}")
print()

# Check relative errors
pytorch_weight_grad_norm = np.linalg.norm(weight_pt.grad.numpy())
relative_error = grad_weight_diff.max() / (pytorch_weight_grad_norm + 1e-8)
print(f"Relative error (max_diff / grad_norm): {relative_error:.2e}")
print()

print("=" * 60)
print("Conclusion:")
print("=" * 60)
if weight_match:
    print("✓ Both gradients pass tolerance")
else:
    print("✗ Weight gradient slightly exceeds tolerance")
    print(f"  Max error: {grad_weight_diff.max():.2e} vs threshold: 1.00e-04")
    print(f"  Exceeds by: {(grad_weight_diff.max() - 1e-4) / 1e-4 * 100:.1f}%")
    if weight_match_relaxed:
        print(f"  Would pass with rtol=3e-4, atol=3e-4")
    print()
    print("This is likely due to floating-point accumulation differences")
    print("across many multiply-add operations in large tensors.")
