import nabla as nb
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available, skipping comparison tests")


def compare_with_pytorch(nabla_result, pytorch_result, test_name, rtol=1e-5, atol=1e-6):
    """Compare Nabla and PyTorch results."""
    nabla_np = nabla_result.to_numpy()
    pytorch_np = pytorch_result.detach().cpu().numpy()
    
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"Nabla shape:   {nabla_np.shape}")
    print(f"PyTorch shape: {pytorch_np.shape}")
    print(f"Nabla output:\n{nabla_np}")
    print(f"PyTorch output:\n{pytorch_np}")
    
    if nabla_np.shape != pytorch_np.shape:
        print(f"❌ FAILED: Shape mismatch!")
        return False
    
    is_close = np.allclose(nabla_np, pytorch_np, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(nabla_np - pytorch_np))
    print(f"Max difference: {max_diff}")
    
    if is_close:
        print(f"✅ PASSED: Results match within tolerance (rtol={rtol}, atol={atol})")
        return True
    else:
        print(f"❌ FAILED: Results differ by more than tolerance")
        print(f"Difference:\n{nabla_np - pytorch_np}")
        return False


def test_basic_fold():
    """Test basic fold operation with default parameters."""
    print("\n" + "="*60)
    print("TEST 1: Basic Fold (stride=1, no padding/dilation)")
    print("="*60)
    
    # For fold operation with output_size=(4,4) and kernel_size=(2,2) and stride=1
    # Number of blocks L = (4 - 2 + 1) * (4 - 2 + 1) = 3 * 3 = 9
    # Input shape: (N, C * kernel_h * kernel_w, L) = (1, 2 * 4, 9) = (1, 8, 9)
    
    # Create input data
    input_data = np.arange(72, dtype=np.float32).reshape(1, 8, 9)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_fold(x):
        return nb.fold(x, (4, 4), (2, 2))
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size=(4, 4), kernel_size=(2, 2), stride=1)
        return compare_with_pytorch(nabla_result, pytorch_result, "Basic Fold")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_fold_with_stride():
    """Test fold operation with stride > 1."""
    print("\n" + "="*60)
    print("TEST 2: Fold with Stride=2")
    print("="*60)
    
    # With stride=2, output_size=(6,6), kernel=(2,2)
    # Number of blocks = ((6-2)/2 + 1) * ((6-2)/2 + 1) = 3 * 3 = 9
    # Input shape: (1, 8, 9)
    
    input_data = np.arange(72, dtype=np.float32).reshape(1, 8, 9)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_fold(x):
        return nb.fold(x, (6, 6), (2, 2), stride=2)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size=(6, 6), kernel_size=(2, 2), stride=2)
        return compare_with_pytorch(nabla_result, pytorch_result, "Fold with Stride=2")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_fold_with_padding():
    """Test fold operation with padding."""
    print("\n" + "="*60)
    print("TEST 3: Fold with Padding=1")
    print("="*60)
    
    # With padding=1, output_size=(4,4), kernel=(2,2), stride=1
    # Number of blocks = ((4+2*1-2)/1 + 1) * ((4+2*1-2)/1 + 1) = 5 * 5 = 25
    # Input shape: (1, 8, 25)
    
    input_data = np.arange(200, dtype=np.float32).reshape(1, 8, 25)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_fold(x):
        return nb.fold(x, (4, 4), (2, 2), stride=1, padding=1)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size=(4, 4), kernel_size=(2, 2), stride=1, padding=1)
        return compare_with_pytorch(nabla_result, pytorch_result, "Fold with Padding=1")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_fold_with_dilation():
    """Test fold operation with dilation."""
    print("\n" + "="*60)
    print("TEST 4: Fold with Dilation=2")
    print("="*60)
    
    # With dilation=2, output_size=(6,6), kernel=(2,2), stride=1
    # Effective kernel size = (2-1)*2 + 1 = 3
    # Number of blocks = ((6-3)/1 + 1) * ((6-3)/1 + 1) = 4 * 4 = 16
    # Input shape: (1, 8, 16)
    
    input_data = np.arange(128, dtype=np.float32).reshape(1, 8, 16)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_fold(x):
        return nb.fold(x, (6, 6), (2, 2), stride=1, dilation=2)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size=(6, 6), kernel_size=(2, 2), stride=1, dilation=2)
        return compare_with_pytorch(nabla_result, pytorch_result, "Fold with Dilation=2")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_fold_combined_params():
    """Test fold operation with combined parameters."""
    print("\n" + "="*60)
    print("TEST 5: Fold with Combined Parameters (stride=2, padding=1, dilation=1)")
    print("="*60)
    
    # With stride=2, padding=1, dilation=1, output_size=(8,8), kernel=(3,3)
    # Number of blocks = ((8+2*1-(3-1)*1-1)/2 + 1) * ((8+2*1-(3-1)*1-1)/2 + 1) 
    #                  = ((8+2-2-1)/2 + 1) * ((8+2-2-1)/2 + 1)
    #                  = (7/2 + 1) * (7/2 + 1) = 4 * 4 = 16
    # Input shape: (1, 18, 16) for 2 channels with 3x3 kernel
    
    input_data = np.arange(288, dtype=np.float32).reshape(1, 18, 16)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_fold(x):
        return nb.fold(x, (8, 8), (3, 3), stride=2, padding=1, dilation=1)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size=(8, 8), kernel_size=(3, 3), stride=2, padding=1, dilation=1)
        return compare_with_pytorch(nabla_result, pytorch_result, "Fold with Combined Parameters")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_larger_batch():
    """Test with larger batch size."""
    print("\n" + "="*60)
    print("TEST 6: Larger Batch Size (N=4)")
    print("="*60)
    
    batch_size = 4
    channels = 3
    kernel_size = (2, 2)
    output_size = (5, 5)
    
    # Calculate input size - with stride=1 (default)
    L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
    input_shape = (batch_size, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size)
        return compare_with_pytorch(nabla_result, pytorch_result, "Larger Batch Size")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_large_kernel():
    """Test with larger kernel size."""
    print("\n" + "="*60)
    print("TEST 7: Large Kernel (5x5)")
    print("="*60)
    
    channels = 2
    kernel_size = (5, 5)
    output_size = (8, 8)
    stride = (1, 1)
    
    # Calculate input size
    L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size, stride=stride)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size, stride=stride)
        return compare_with_pytorch(nabla_result, pytorch_result, "Large Kernel")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_non_square_kernel():
    """Test with non-square kernel."""
    print("\n" + "="*60)
    print("TEST 8: Non-Square Kernel (3x5)")
    print("="*60)
    
    channels = 2
    kernel_size = (3, 5)
    output_size = (6, 8)
    stride = (2, 2)
    
    # Calculate input size
    L = ((output_size[0] - kernel_size[0]) // stride[0] + 1) * \
        ((output_size[1] - kernel_size[1]) // stride[1] + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size, stride=stride)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size, stride=stride)
        return compare_with_pytorch(nabla_result, pytorch_result, "Non-Square Kernel")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_non_square_output():
    """Test with non-square output."""
    print("\n" + "="*60)
    print("TEST 9: Non-Square Output (10x6)")
    print("="*60)
    
    channels = 3
    kernel_size = (2, 3)
    output_size = (10, 6)
    stride = (1, 1)
    
    # Calculate input size
    L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size, stride=stride)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size, stride=stride)
        return compare_with_pytorch(nabla_result, pytorch_result, "Non-Square Output")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_large_dilation():
    """Test with larger dilation."""
    print("\n" + "="*60)
    print("TEST 10: Large Dilation (3x3)")
    print("="*60)
    
    channels = 2
    kernel_size = (2, 2)
    output_size = (12, 12)
    dilation = (3, 3)
    stride = (1, 1)
    
    # Calculate input size with dilation
    effective_kernel_h = (kernel_size[0] - 1) * dilation[0] + 1
    effective_kernel_w = (kernel_size[1] - 1) * dilation[1] + 1
    L = (output_size[0] - effective_kernel_h + 1) * (output_size[1] - effective_kernel_w + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size, dilation=dilation, stride=stride)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size, dilation=dilation, stride=stride)
        return compare_with_pytorch(nabla_result, pytorch_result, "Large Dilation")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_asymmetric_stride():
    """Test with asymmetric stride."""
    print("\n" + "="*60)
    print("TEST 11: Asymmetric Stride (1x3)")
    print("="*60)
    
    channels = 2
    kernel_size = (2, 2)
    output_size = (8, 9)
    stride = (1, 3)
    
    # Calculate input size
    L = ((output_size[0] - kernel_size[0]) // stride[0] + 1) * \
        ((output_size[1] - kernel_size[1]) // stride[1] + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size, stride=stride)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size, stride=stride)
        return compare_with_pytorch(nabla_result, pytorch_result, "Asymmetric Stride")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_asymmetric_padding():
    """Test with asymmetric padding (different for height and width)."""
    print("\n" + "="*60)
    print("TEST 12: Asymmetric Padding (2x1)")
    print("="*60)
    
    channels = 2
    kernel_size = (3, 3)
    output_size = (6, 6)
    padding = (2, 1)
    stride = (1, 1)
    
    # Calculate input size with padding
    padded_h = output_size[0] + 2 * padding[0]
    padded_w = output_size[1] + 2 * padding[1]
    L = (padded_h - kernel_size[0] + 1) * (padded_w - kernel_size[1] + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size, padding=padding, stride=stride)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size, padding=padding, stride=stride)
        return compare_with_pytorch(nabla_result, pytorch_result, "Asymmetric Padding")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_complex_combination():
    """Test with complex combination of all parameters."""
    print("\n" + "="*60)
    print("TEST 13: Complex Combination")
    print("="*60)
    
    channels = 4
    kernel_size = (3, 4)
    output_size = (14, 16)
    stride = (2, 3)
    padding = (1, 2)
    dilation = (2, 1)
    
    # Calculate input size with all parameters
    padded_h = output_size[0] + 2 * padding[0]
    padded_w = output_size[1] + 2 * padding[1]
    effective_kernel_h = (kernel_size[0] - 1) * dilation[0] + 1
    effective_kernel_w = (kernel_size[1] - 1) * dilation[1] + 1
    L = ((padded_h - effective_kernel_h) // stride[0] + 1) * \
        ((padded_w - effective_kernel_w) // stride[1] + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size, 
                                            stride=stride, padding=padding, dilation=dilation)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        return compare_with_pytorch(nabla_result, pytorch_result, "Complex Combination")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_single_channel():
    """Test with single channel."""
    print("\n" + "="*60)
    print("TEST 14: Single Channel")
    print("="*60)
    
    channels = 1
    kernel_size = (3, 3)
    output_size = (7, 7)
    
    # Calculate input size
    L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size)
        return compare_with_pytorch(nabla_result, pytorch_result, "Single Channel")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_many_channels():
    """Test with many channels."""
    print("\n" + "="*60)
    print("TEST 15: Many Channels (16)")
    print("="*60)
    
    channels = 16
    kernel_size = (2, 2)
    output_size = (5, 5)
    
    # Calculate input size
    L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
    input_shape = (1, channels * kernel_size[0] * kernel_size[1], L)
    
    print(f"Input shape: {input_shape}")
    
    # Create input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    nabla_input = nb.tensor(input_data)
    
    # Nabla fold
    def nabla_fold(x):
        return nb.fold(x, output_size, kernel_size)
    jitted_nabla_fold = nb.jit(nabla_fold)
    nabla_result = jitted_nabla_fold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.fold(torch_input, output_size, kernel_size)
        return compare_with_pytorch(nabla_result, pytorch_result, "Many Channels")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


if __name__ == "__main__":
    if not PYTORCH_AVAILABLE:
        print("\n⚠️  WARNING: PyTorch is not installed. Comparison tests will be skipped.")
        print("To run comparison tests, install PyTorch: pip install torch\n")
    
    results = []
    
    # Original tests
    results.append(("Basic Fold", test_basic_fold()))
    results.append(("Fold with Stride", test_fold_with_stride()))
    results.append(("Fold with Padding", test_fold_with_padding()))
    results.append(("Fold with Dilation", test_fold_with_dilation()))
    results.append(("Fold with Combined Parameters", test_fold_combined_params()))
    
    # Rigorous tests
    results.append(("Larger Batch Size", test_larger_batch()))
    results.append(("Large Kernel", test_large_kernel()))
    results.append(("Non-Square Kernel", test_non_square_kernel()))
    results.append(("Non-Square Output", test_non_square_output()))
    results.append(("Large Dilation", test_large_dilation()))
    results.append(("Asymmetric Stride", test_asymmetric_stride()))
    results.append(("Asymmetric Padding", test_asymmetric_padding()))
    results.append(("Complex Combination", test_complex_combination()))
    results.append(("Single Channel", test_single_channel()))
    results.append(("Many Channels", test_many_channels()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed!")
