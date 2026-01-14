#!/usr/bin/env python
"""Test script for Conv2D Transpose VJP (backward pass) implementation."""

import numpy as np
import nabla as nb

try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - skipping tests")
    exit(0)


def test_conv2d_transpose_vjp_basic():
    """Test basic VJP for conv2d_transpose with simple configuration."""
    print("\n=== Testing Conv2D Transpose VJP Basic ===")
    
    np.random.seed(42)
    
    # Simple configuration
    # For transpose conv: input (N, C_in, H, W), weight (C_in, C_out, K, K)
    input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
    weight_np = np.random.randn(3, 16, 3, 3).astype(np.float32)
    
    # PyTorch reference
    input_pt = torch.from_numpy(input_np).requires_grad_(True)
    weight_pt = torch.from_numpy(weight_np).requires_grad_(True)
    
    output_pt = F.conv_transpose2d(input_pt, weight_pt)
    grad_output_np = np.random.randn(*output_pt.shape).astype(np.float32)
    grad_output_pt = torch.from_numpy(grad_output_np)
    output_pt.backward(grad_output_pt)
    
    # Nabla VJP using nb.vjp
    input_nb = nb.tensor(input_np)
    weight_nb = nb.tensor(weight_np)
    grad_output_nb = nb.tensor(grad_output_np)
    
    try:
        # Define function and compute VJP
        def conv_transpose_fn(inp, weight):
            return nb.conv2d_transpose(inp, weight)
        
        output_nb, vjp_fn = nb.vjp(conv_transpose_fn, input_nb, weight_nb)
        grad_input_nb, grad_weight_nb = vjp_fn(grad_output_nb)
        
        # Compare
        grad_input_match = np.allclose(grad_input_nb.to_numpy(), input_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
        grad_weight_match = np.allclose(grad_weight_nb.to_numpy(), weight_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
        
        print(f"✓ Gradient w.r.t. input matches PyTorch: {grad_input_match}")
        print(f"✓ Gradient w.r.t. weight matches PyTorch: {grad_weight_match}")
        
        if not grad_input_match:
            print(f"  Max diff (input): {np.max(np.abs(grad_input_nb.to_numpy() - input_pt.grad.numpy()))}")
            print(f"  Input grad shape: Nabla={grad_input_nb.shape}, PyTorch={input_pt.grad.shape}")
        if not grad_weight_match:
            print(f"  Max diff (weight): {np.max(np.abs(grad_weight_nb.to_numpy() - weight_pt.grad.numpy()))}")
            print(f"  Weight grad shape: Nabla={grad_weight_nb.shape}, PyTorch={weight_pt.grad.shape}")
        
        return grad_input_match and grad_weight_match
    except Exception as e:
        print(f"✗ VJP failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conv2d_transpose_vjp_with_stride():
    """Test VJP with stride."""
    print("\n=== Testing Conv2D Transpose VJP with Stride ===")
    
    np.random.seed(43)
    
    input_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    weight_np = np.random.randn(3, 8, 3, 3).astype(np.float32)
    
    # PyTorch reference
    input_pt = torch.from_numpy(input_np).requires_grad_(True)
    weight_pt = torch.from_numpy(weight_np).requires_grad_(True)
    
    output_pt = F.conv_transpose2d(input_pt, weight_pt, stride=2)
    grad_output_pt = torch.randn_like(output_pt)
    output_pt.backward(grad_output_pt)
    
    # Nabla VJP
    input_nb = nb.tensor(input_np)
    weight_nb = nb.tensor(weight_np)
    grad_output_nb = nb.tensor(grad_output_pt.numpy())
    
    try:
        def conv_transpose_fn(inp, weight):
            return nb.conv2d_transpose(inp, weight, stride=2)
        
        output_nb, vjp_fn = nb.vjp(conv_transpose_fn, input_nb, weight_nb)
        grad_input_nb, grad_weight_nb = vjp_fn(grad_output_nb)
        
        # Compare
        grad_input_match = np.allclose(grad_input_nb.to_numpy(), input_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
        grad_weight_match = np.allclose(grad_weight_nb.to_numpy(), weight_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
        
        print(f"✓ Gradient w.r.t. input matches PyTorch: {grad_input_match}")
        print(f"✓ Gradient w.r.t. weight matches PyTorch: {grad_weight_match}")
        
        if not grad_input_match:
            print(f"  Max diff (input): {np.max(np.abs(grad_input_nb.to_numpy() - input_pt.grad.numpy()))}")
        if not grad_weight_match:
            print(f"  Max diff (weight): {np.max(np.abs(grad_weight_nb.to_numpy() - weight_pt.grad.numpy()))}")
        
        return grad_input_match and grad_weight_match
    except Exception as e:
        print(f"✗ VJP failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conv2d_transpose_vjp_with_padding():
    """Test VJP with padding."""
    print("\n=== Testing Conv2D Transpose VJP with Padding ===")
    
    np.random.seed(44)
    
    input_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    weight_np = np.random.randn(3, 8, 3, 3).astype(np.float32)
    
    # PyTorch reference
    input_pt = torch.from_numpy(input_np).requires_grad_(True)
    weight_pt = torch.from_numpy(weight_np).requires_grad_(True)
    
    output_pt = F.conv_transpose2d(input_pt, weight_pt, padding=1)
    grad_output_pt = torch.randn_like(output_pt)
    output_pt.backward(grad_output_pt)
    
    # Nabla VJP
    input_nb = nb.tensor(input_np)
    weight_nb = nb.tensor(weight_np)
    grad_output_nb = nb.tensor(grad_output_pt.numpy())
    
    try:
        def conv_transpose_fn(inp, weight):
            return nb.conv2d_transpose(inp, weight, padding=1)
        
        output_nb, vjp_fn = nb.vjp(conv_transpose_fn, input_nb, weight_nb)
        grad_input_nb, grad_weight_nb = vjp_fn(grad_output_nb)
        
        # Compare
        grad_input_match = np.allclose(grad_input_nb.to_numpy(), input_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
        grad_weight_match = np.allclose(grad_weight_nb.to_numpy(), weight_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
        
        print(f"✓ Gradient w.r.t. input matches PyTorch: {grad_input_match}")
        print(f"✓ Gradient w.r.t. weight matches PyTorch: {grad_weight_match}")
        
        if not grad_input_match:
            print(f"  Max diff (input): {np.max(np.abs(grad_input_nb.to_numpy() - input_pt.grad.numpy()))}")
        if not grad_weight_match:
            print(f"  Max diff (weight): {np.max(np.abs(grad_weight_nb.to_numpy() - weight_pt.grad.numpy()))}")
        
        return grad_input_match and grad_weight_match
    except Exception as e:
        print(f"✗ VJP failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conv2d_transpose_vjp_comprehensive():
    """Test VJP with many different configurations."""
    print("\n=== Testing Conv2D Transpose VJP Comprehensive (Multiple Configs) ===")
    
    # Test configurations: (input_shape, weight_shape, stride, padding, name, seed)
    configs = [
        # Basic configurations
        ((1, 1, 5, 5), (1, 1, 3, 3), 1, 0, "basic_1x1", 100),
        ((2, 3, 8, 8), (3, 16, 3, 3), 1, 0, "basic_multi_ch", 101),
        
        # Different strides
        ((2, 3, 8, 8), (3, 8, 3, 3), 2, 0, "stride_2", 102),
        ((1, 4, 10, 10), (4, 16, 5, 5), 3, 0, "stride_3", 103),
        ((2, 3, 8, 8), (3, 8, 3, 3), (2, 1), 0, "stride_(2,1)", 104),
        ((2, 3, 8, 8), (3, 8, 3, 3), (1, 2), 0, "stride_(1,2)", 105),
        
        # Different padding
        ((2, 3, 8, 8), (3, 8, 3, 3), 1, 1, "padding_1", 106),
        ((2, 3, 8, 8), (3, 8, 3, 3), 1, 2, "padding_2", 107),
        ((2, 3, 8, 8), (3, 8, 3, 3), 1, (1, 2), "padding_(1,2)", 108),
        ((2, 3, 8, 8), (3, 8, 3, 3), 1, (2, 1), "padding_(2,1)", 109),
        
        # Combined stride and padding
        ((2, 3, 8, 8), (3, 8, 3, 3), 2, 1, "stride_2_pad_1", 110),
        ((1, 4, 10, 10), (4, 16, 5, 5), 2, 2, "stride_2_pad_2", 111),
        
        # Different kernel sizes
        ((2, 3, 8, 8), (3, 8, 5, 5), 1, 0, "kernel_5x5", 112),
        ((2, 3, 8, 8), (3, 8, 7, 7), 1, 0, "kernel_7x7", 113),
        ((1, 4, 10, 10), (4, 16, 4, 4), 1, 0, "kernel_4x4_even", 114),
        
        # Different batch sizes
        ((1, 3, 8, 8), (3, 8, 3, 3), 1, 1, "batch_1", 115),
        ((4, 3, 8, 8), (3, 8, 3, 3), 2, 1, "batch_4", 116),
        ((8, 3, 8, 8), (3, 8, 3, 3), 1, 1, "batch_8", 117),
        
        # Different channel counts
        ((2, 1, 8, 8), (1, 8, 3, 3), 1, 1, "1_input_channel", 118),
        ((2, 16, 8, 8), (16, 32, 3, 3), 1, 1, "16_input_channels", 119),
        ((2, 32, 8, 8), (32, 64, 3, 3), 2, 1, "32_input_channels", 120),
        
        # Rectangular inputs
        ((2, 3, 8, 12), (3, 8, 3, 3), 1, 1, "rect_8x12", 121),
        ((1, 4, 6, 9), (4, 16, 3, 3), 2, 1, "rect_6x9_stride2", 122),
        
        # Edge cases
        ((1, 3, 3, 3), (3, 8, 3, 3), 1, 1, "small_3x3", 123),
        ((2, 4, 7, 7), (4, 12, 4, 4), 2, 1, "odd_7x7", 124),
    ]
    
    results = []
    for input_shape, weight_shape, stride, padding, name, seed in configs:
        np.random.seed(seed)
        
        input_np = np.random.randn(*input_shape).astype(np.float32)
        weight_np = np.random.randn(*weight_shape).astype(np.float32)
        
        # PyTorch reference
        input_pt = torch.from_numpy(input_np).requires_grad_(True)
        weight_pt = torch.from_numpy(weight_np).requires_grad_(True)
        
        try:
            output_pt = F.conv_transpose2d(input_pt, weight_pt, stride=stride, padding=padding)
            grad_output_pt = torch.randn_like(output_pt)
            output_pt.backward(grad_output_pt)
            
            # Nabla VJP
            input_nb = nb.tensor(input_np)
            weight_nb = nb.tensor(weight_np)
            grad_output_nb = nb.tensor(grad_output_pt.numpy())
            
            def conv_transpose_fn(inp, weight):
                return nb.conv2d_transpose(inp, weight, stride=stride, padding=padding)
            
            output_nb, vjp_fn = nb.vjp(conv_transpose_fn, input_nb, weight_nb)
            grad_input_nb, grad_weight_nb = vjp_fn(grad_output_nb)
            
            # Compare
            grad_input_match = np.allclose(grad_input_nb.to_numpy(), input_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
            grad_weight_match = np.allclose(grad_weight_nb.to_numpy(), weight_pt.grad.numpy(), rtol=1e-4, atol=1e-4)
            
            if grad_input_match and grad_weight_match:
                print(f"  ✓ {name}")
                results.append(True)
            else:
                print(f"  ✗ {name}: FAILED")
                if not grad_input_match:
                    print(f"    Input grad max diff: {np.max(np.abs(grad_input_nb.to_numpy() - input_pt.grad.numpy())):.2e}")
                if not grad_weight_match:
                    print(f"    Weight grad max diff: {np.max(np.abs(grad_weight_nb.to_numpy() - weight_pt.grad.numpy())):.2e}")
                results.append(False)
                
        except Exception as e:
            print(f"  ✗ {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\n  Summary: {passed}/{total} configurations passed")
    return all(results)


if __name__ == '__main__':
    print("=" * 60)
    print("CONV2D TRANSPOSE VJP (BACKWARD PASS) TEST SUITE")
    print("=" * 60)
    
    results = []
    results.append(test_conv2d_transpose_vjp_basic())
    results.append(test_conv2d_transpose_vjp_with_stride())
    results.append(test_conv2d_transpose_vjp_with_padding())
    results.append(test_conv2d_transpose_vjp_comprehensive())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"OVERALL RESULTS: {passed}/{total} test groups passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All VJP tests passed!")
        exit(0)
    else:
        print("✗ Some VJP tests failed")
        exit(1)
