#!/usr/bin/env python
"""Test script for Conv2D VJP (backward pass) implementation."""

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

def test_conv2d_vjp_basic():
    """Test basic VJP for conv2d with simple configuration."""
    print("\n=== Testing Conv2D VJP Basic ===")
    
    np.random.seed(42)
    
    # Simple configuration
    input_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    weight_np = np.random.randn(16, 3, 3, 3).astype(np.float32)
    grad_output_np = np.random.randn(2, 16, 6, 6).astype(np.float32)
    
    # PyTorch reference
    input_pt = torch.from_numpy(input_np).requires_grad_(True)
    weight_pt = torch.from_numpy(weight_np).requires_grad_(True)
    grad_output_pt = torch.from_numpy(grad_output_np)
    
    output_pt = F.conv2d(input_pt, weight_pt)
    output_pt.backward(grad_output_pt)
    
    # Nabla VJP using nb.vjp
    input_nb = nb.tensor(input_np)
    weight_nb = nb.tensor(weight_np)
    grad_output_nb = nb.tensor(grad_output_np)
    
    try:
        # Define function and compute VJP
        def conv_fn(inp, weight):
            return nb.conv2d(inp, weight)
        
        output_nb, vjp_fn = nb.vjp(conv_fn, input_nb, weight_nb)
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


def test_conv2d_vjp_with_padding():
    """Test VJP with padding."""
    print("\n=== Testing Conv2D VJP with Padding ===")
    
    np.random.seed(43)
    
    input_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    weight_np = np.random.randn(8, 3, 3, 3).astype(np.float32)
    
    # PyTorch reference
    input_pt = torch.from_numpy(input_np).requires_grad_(True)
    weight_pt = torch.from_numpy(weight_np).requires_grad_(True)
    
    output_pt = F.conv2d(input_pt, weight_pt, padding=1)
    grad_output_pt = torch.randn_like(output_pt)
    output_pt.backward(grad_output_pt)
    
    # Nabla VJP
    input_nb = nb.tensor(input_np)
    weight_nb = nb.tensor(weight_np)
    grad_output_nb = nb.tensor(grad_output_pt.numpy())
    
    try:
        def conv_fn(inp, weight):
            return nb.conv2d(inp, weight, padding=1)
        
        output_nb, vjp_fn = nb.vjp(conv_fn, input_nb, weight_nb)
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


def test_conv2d_vjp_with_stride():
    """Test VJP with stride."""
    print("\n=== Testing Conv2D VJP with Stride ===")
    
    np.random.seed(44)
    
    input_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
    weight_np = np.random.randn(8, 3, 3, 3).astype(np.float32)
    
    # PyTorch reference
    input_pt = torch.from_numpy(input_np).requires_grad_(True)
    weight_pt = torch.from_numpy(weight_np).requires_grad_(True)
    
    output_pt = F.conv2d(input_pt, weight_pt, stride=2)
    grad_output_pt = torch.randn_like(output_pt)
    output_pt.backward(grad_output_pt)
    
    # Nabla VJP
    input_nb = nb.tensor(input_np)
    weight_nb = nb.tensor(weight_np)
    grad_output_nb = nb.tensor(grad_output_pt.numpy())
    
    try:
        def conv_fn(inp, weight):
            return nb.conv2d(inp, weight, stride=2)
        
        output_nb, vjp_fn = nb.vjp(conv_fn, input_nb, weight_nb)
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


def test_conv2d_vjp_comprehensive():
    """Test VJP with many different configurations."""
    print("\n=== Testing Conv2D VJP Comprehensive (Multiple Configs) ===")
    
    # Test configurations: (input_shape, weight_shape, stride, padding, name, seed)
    configs = [
        # Different strides
        ((2, 3, 16, 16), (8, 3, 3, 3), 3, 0, "stride_3", 100),
        ((1, 4, 20, 20), (16, 4, 5, 5), 2, 0, "stride_2_5x5_kernel", 101),
        ((2, 3, 16, 16), (8, 3, 3, 3), (2, 1), 0, "stride_(2,1)", 102),
        ((2, 3, 16, 16), (8, 3, 3, 3), (1, 2), 0, "stride_(1,2)", 103),
        
        # Different padding
        ((2, 3, 16, 16), (8, 3, 3, 3), 1, 2, "padding_2", 104),
        ((2, 3, 16, 16), (8, 3, 3, 3), 1, (1, 2), "padding_(1,2)", 105),
        ((2, 3, 16, 16), (8, 3, 3, 3), 1, (2, 1), "padding_(2,1)", 106),
        ((1, 4, 12, 12), (16, 4, 5, 5), 1, 2, "padding_2_5x5", 107),
        
        # Combined stride and padding
        ((2, 3, 16, 16), (8, 3, 3, 3), 2, 1, "stride_2_padding_1", 108),
        ((1, 4, 20, 20), (16, 4, 5, 5), 2, 2, "stride_2_padding_2_5x5", 109),
        ((2, 3, 32, 32), (16, 3, 3, 3), (2, 1), (1, 2), "aniso_stride_pad", 110),
        ((1, 8, 24, 24), (32, 8, 7, 7), 3, 3, "stride_3_pad_3_7x7", 111),
        
        # Different input sizes
        ((1, 3, 32, 32), (16, 3, 3, 3), 1, 1, "32x32_input", 112),
        ((2, 3, 64, 64), (8, 3, 3, 3), 2, 1, "64x64_input_stride2", 113),
        ((1, 4, 10, 10), (12, 4, 3, 3), 1, 1, "10x10_input", 114),
        
        # Different kernel sizes
        ((2, 3, 16, 16), (8, 3, 5, 5), 1, 0, "5x5_kernel", 115),
        ((2, 3, 16, 16), (8, 3, 7, 7), 1, 0, "7x7_kernel", 116),
        ((1, 4, 20, 20), (16, 4, 4, 4), 1, 0, "4x4_kernel_even", 117),
        
        # Different batch sizes
        ((1, 3, 16, 16), (8, 3, 3, 3), 1, 1, "batch_1", 118),
        ((4, 3, 16, 16), (8, 3, 3, 3), 2, 1, "batch_4", 119),
        ((8, 3, 16, 16), (8, 3, 3, 3), 1, 1, "batch_8", 120),
        
        # Different channel counts
        ((2, 1, 16, 16), (8, 1, 3, 3), 1, 1, "1_input_channel", 121),
        ((2, 16, 16, 16), (32, 16, 3, 3), 1, 1, "16_input_channels", 122),
        ((2, 32, 16, 16), (64, 32, 3, 3), 2, 1, "32_input_channels", 123),
        
        # Rectangular inputs
        ((2, 3, 16, 24), (8, 3, 3, 3), 1, 1, "rect_16x24", 124),
        ((1, 4, 12, 18), (16, 4, 3, 3), 2, 1, "rect_12x18_stride2", 125),
        ((2, 3, 10, 15), (8, 3, 3, 3), 1, 0, "rect_10x15", 126),
        
        # Edge cases
        ((1, 3, 8, 8), (8, 3, 3, 3), 1, "same", "padding_same", 127),
        ((2, 3, 8, 8), (8, 3, 3, 3), 1, "valid", "padding_valid", 128),
        ((1, 3, 7, 7), (8, 3, 3, 3), 1, 1, "odd_7x7", 129),
        ((2, 4, 15, 15), (12, 4, 4, 4), 2, 1, "odd_15x15_even_kernel", 130),
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
            output_pt = F.conv2d(input_pt, weight_pt, stride=stride, padding=padding)
            grad_output_pt = torch.randn_like(output_pt)
            output_pt.backward(grad_output_pt)
            
            # Nabla VJP
            input_nb = nb.tensor(input_np)
            weight_nb = nb.tensor(weight_np)
            grad_output_nb = nb.tensor(grad_output_pt.numpy())
            
            def conv_fn(inp, weight):
                return nb.conv2d(inp, weight, stride=stride, padding=padding)
            
            output_nb, vjp_fn = nb.vjp(conv_fn, input_nb, weight_nb)
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
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\n  Summary: {passed}/{total} configurations passed")
    return all(results)


def test_conv2d_vjp_real_world():
    """
    Test VJP with configurations inspired by real-world CNN architectures.
    
    This covers typical patterns from:
    - VGG-style networks (stacked 3x3 convs)
    - ResNet bottleneck blocks (1x1, 3x3, 1x1 sequences)
    - EfficientNet (larger kernels, expansion patterns)
    - Feature pyramids (multi-scale processing)
    - Production scenarios (large batches, high channels)
    """
    print("\n=== Testing Conv2D VJP Real-World Scenarios ===")
    
    np.random.seed(42)
    
    # Configuration format: (name, batch, in_ch, out_ch, H, W, K, stride, padding, description)
    real_world_configs = [
        # === VGG-Style Early Layers ===
        ("VGG_early_layer_224", 16, 3, 64, 224, 224, 3, 1, 1,
         "VGG early layer: RGB input → 64 channels"),
        ("VGG_early_layer_112", 16, 64, 64, 112, 112, 3, 1, 1,
         "VGG early layer: 64→64 channels after pooling"),
        
        # === VGG-Style Mid Layers ===
        ("VGG_mid_128ch", 16, 64, 128, 56, 56, 3, 1, 1,
         "VGG mid layer: 64→128 channel expansion"),
        ("VGG_mid_256ch", 16, 128, 256, 28, 28, 3, 1, 1,
         "VGG mid layer: 128→256 channel expansion"),
        
        # === VGG-Style Late Layers ===
        ("VGG_late_512ch", 16, 256, 512, 14, 14, 3, 1, 1,
         "VGG late layer: 256→512 channels"),
        ("VGG_final_512ch", 16, 512, 512, 7, 7, 3, 1, 1,
         "VGG final layer: 512→512 before FC"),
        
        # === ResNet Bottleneck: Compression (1x1) ===
        ("ResNet_bottleneck_compress", 32, 256, 64, 56, 56, 1, 1, 0,
         "ResNet bottleneck: 256→64 compression with 1x1"),
        ("ResNet_bottleneck_compress_deep", 32, 1024, 256, 14, 14, 1, 1, 0,
         "ResNet deep bottleneck: 1024→256 compression"),
        
        # === ResNet Bottleneck: Main Conv (3x3) ===
        ("ResNet_bottleneck_main", 32, 64, 64, 56, 56, 3, 1, 1,
         "ResNet bottleneck: 3x3 at reduced dims"),
        ("ResNet_bottleneck_main_stride2", 32, 64, 64, 56, 56, 3, 2, 1,
         "ResNet bottleneck: 3x3 with stride-2 downsampling"),
        
        # === ResNet Bottleneck: Expansion (1x1) ===
        ("ResNet_bottleneck_expand", 32, 64, 256, 56, 56, 1, 1, 0,
         "ResNet bottleneck: 64→256 expansion with 1x1"),
        ("ResNet_bottleneck_expand_deep", 32, 512, 2048, 7, 7, 1, 1, 0,
         "ResNet deep bottleneck: 512→2048 expansion"),
        
        # === ResNet Downsampling Shortcut ===
        ("ResNet_downsample_56to28", 32, 64, 128, 56, 56, 1, 2, 0,
         "ResNet shortcut: 1x1 stride-2 downsampling"),
        ("ResNet_downsample_28to14", 32, 128, 256, 28, 28, 1, 2, 0,
         "ResNet shortcut: 1x1 stride-2 downsampling"),
        
        # === EfficientNet-Style: Larger Kernels ===
        ("EfficientNet_5x5_early", 16, 32, 64, 112, 112, 5, 1, 2,
         "EfficientNet: 5x5 kernel early stage"),
        ("EfficientNet_5x5_mid", 16, 96, 192, 28, 28, 5, 1, 2,
         "EfficientNet: 5x5 kernel mid stage"),
        
        # === EfficientNet-Style: Expansion + Large Kernel ===
        ("EfficientNet_expansion", 16, 40, 240, 56, 56, 1, 1, 0,
         "EfficientNet: 1x1 expansion (6x channels)"),
        ("EfficientNet_7x7_stride2", 16, 240, 240, 56, 56, 7, 2, 3,
         "EfficientNet: 7x7 depthwise-like with stride-2"),
        
        # === Feature Pyramid Network: Multi-Scale ===
        ("FPN_P3_lateral", 8, 256, 256, 56, 56, 1, 1, 0,
         "FPN: P3 lateral connection (1x1)"),
        ("FPN_P4_lateral", 8, 512, 256, 28, 28, 1, 1, 0,
         "FPN: P4 lateral connection (1x1)"),
        ("FPN_smooth_3x3", 8, 256, 256, 56, 56, 3, 1, 1,
         "FPN: 3x3 smoothing convolution"),
        
        # === Production Large Batch Scenarios ===
        ("Production_batch64_small", 64, 128, 256, 14, 14, 3, 1, 1,
         "Production: batch=64, small spatial dims"),
        ("Production_batch128_tiny", 128, 256, 512, 7, 7, 3, 1, 1,
         "Production: batch=128, tiny spatial dims"),
        
        # === Very High Channel Counts ===
        ("HighChannel_1024", 8, 512, 1024, 14, 14, 1, 1, 0,
         "High channel: 512→1024 with 1x1"),
        ("HighChannel_2048", 8, 1024, 2048, 7, 7, 1, 1, 0,
         "Very high channel: 1024→2048 with 1x1"),
        
        # === Small Spatial Dimensions (Late Network) ===
        ("SmallSpatial_7x7", 32, 512, 512, 7, 7, 3, 1, 1,
         "Small spatial: 7x7 feature maps"),
        ("SmallSpatial_3x3", 32, 512, 512, 3, 3, 3, 1, 1,
         "Very small spatial: 3x3 feature maps"),
        
        # === Asymmetric Downsampling ===
        ("Asymmetric_224to112", 16, 64, 128, 224, 224, 3, 2, 1,
         "Asymmetric: Large input stride-2 downsampling"),
        ("Asymmetric_56to28", 32, 128, 256, 56, 56, 3, 2, 1,
         "Asymmetric: Mid-network stride-2 downsampling"),
        
        # === Initial Conv (Like ResNet/ImageNet Models) ===
        ("ImageNet_initial_7x7", 16, 3, 64, 224, 224, 7, 2, 3,
         "ImageNet initial: 7x7 stride-2 (ResNet style)"),
        ("ImageNet_initial_3x3", 16, 3, 64, 224, 224, 3, 2, 1,
         "ImageNet initial: 3x3 stride-2 (simpler)"),
        
        # === Pointwise Heavy (Inverted Residuals) ===
        ("InvertedRes_expand", 32, 24, 144, 56, 56, 1, 1, 0,
         "Inverted residual: 1x1 expansion (6x)"),
        ("InvertedRes_project", 32, 144, 24, 56, 56, 1, 1, 0,
         "Inverted residual: 1x1 projection back"),
        
        # === Mixed Patterns ===
        ("Mixed_large_kernel_high_ch", 16, 256, 512, 28, 28, 5, 2, 2,
         "Mixed: Large kernel + high channels + stride-2"),
        ("Mixed_extreme_channels", 8, 2048, 1024, 7, 7, 1, 1, 0,
         "Mixed: Extreme channel reduction in late network"),
        
        # === Edge Cases in Production ===
        ("Edge_minimal_1x1_output", 16, 512, 512, 7, 7, 7, 1, 0,
         "Edge: 7x7 kernel on 7x7 input → 1x1 output"),
        ("Edge_rectangular_input", 16, 128, 256, 112, 224, 3, 2, 1,
         "Edge: Rectangular input (portrait orientation)"),
    ]
    
    passed_configs = []
    failed_configs = []
    
    for config in real_world_configs:
        name, batch, in_ch, out_ch, H, W, K, stride, padding, description = config
        
        try:
            # Create inputs
            input_np = np.random.randn(batch, in_ch, H, W).astype(np.float32) * 0.1
            weight_np = np.random.randn(out_ch, in_ch, K, K).astype(np.float32) * 0.1
            
            # Compute output shape for grad_output
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
            
            # Compare with slightly relaxed tolerance for large real-world tensors
            # Large tensors (e.g., 16×32×64×112×112×5×5) can accumulate more FP error
            grad_input_match = np.allclose(grad_input_nb.to_numpy(), input_pt.grad.numpy(), rtol=3e-4, atol=3e-4)
            grad_weight_match = np.allclose(grad_weight_nb.to_numpy(), weight_pt.grad.numpy(), rtol=3e-4, atol=3e-4)
            
            if grad_input_match and grad_weight_match:
                passed_configs.append(name)
                print(f"  ✓ {name}")
            else:
                failed_configs.append(name)
                print(f"  ✗ {name}")
                if not grad_input_match:
                    err = np.abs(grad_input_nb.to_numpy() - input_pt.grad.numpy()).max()
                    print(f"    Input grad error: {err:.2e}")
                if not grad_weight_match:
                    err = np.abs(grad_weight_nb.to_numpy() - weight_pt.grad.numpy()).max()
                    print(f"    Weight grad error: {err:.2e}")
        
        except Exception as e:
            failed_configs.append(name)
            print(f"  ✗ {name}: {str(e)}")
    
    print(f"\n  Summary: {len(passed_configs)}/{len(real_world_configs)} real-world configurations passed")
    if failed_configs:
        print(f"  Failed: {', '.join(failed_configs)}")
    
    return len(failed_configs) == 0


if __name__ == '__main__':
    print("=" * 60)
    print("CONV2D VJP (BACKWARD PASS) TEST SUITE")
    print("=" * 60)
    
    results = []
    results.append(test_conv2d_vjp_basic())
    results.append(test_conv2d_vjp_with_padding())
    results.append(test_conv2d_vjp_with_stride())
    results.append(test_conv2d_vjp_comprehensive())
    results.append(test_conv2d_vjp_real_world())
    
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
