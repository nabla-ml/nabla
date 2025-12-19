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
    
    print("\n" + "="*60)
    print(f"Test: {test_name}")
    print("="*60)
    print(f"Nabla shape:   {nabla_np.shape}")
    print(f"PyTorch shape: {pytorch_np.shape}")
    
    # Check shapes match
    if nabla_np.shape != pytorch_np.shape:
        print(f"❌ FAILED: Shape mismatch!")
        print(f"  Nabla:   {nabla_np.shape}")
        print(f"  PyTorch: {pytorch_np.shape}")
        return False
    
    # Show sample values
    print("Nabla output:")
    print(nabla_np)
    print("PyTorch output:")
    print(pytorch_np)
    
    # Compare values
    if np.allclose(nabla_np, pytorch_np, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(nabla_np - pytorch_np))
        print(f"Max difference: {max_diff}")
        print(f"✅ PASSED: Results match within tolerance (rtol={rtol}, atol={atol})")
        return True
    else:
        max_diff = np.max(np.abs(nabla_np - pytorch_np))
        print(f"❌ FAILED: Results do not match!")
        print(f"Max difference: {max_diff}")
        print(f"Tolerance: rtol={rtol}, atol={atol}")
        return False


def test_basic_unfold():
    """Test basic unfold operation with default parameters."""
    print("\n" + "="*60)
    print("TEST 1: Basic Unfold (stride=1, no padding/dilation)")
    print("="*60)
    
    # Input shape: (N, C, H, W) = (1, 2, 4, 4)
    # With kernel_size=(2,2), stride=1
    # Number of blocks L = (4 - 2 + 1) * (4 - 2 + 1) = 3 * 3 = 9
    # Output shape: (1, 2 * 2 * 2, 9) = (1, 8, 9)
    
    # Create input data
    input_data = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_unfold(x):
        return nb.unfold(x, (2, 2))
    jitted_nabla_unfold = nb.jit(nabla_unfold)
    nabla_result = jitted_nabla_unfold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.unfold(torch_input, kernel_size=(2, 2), stride=1)
        return compare_with_pytorch(nabla_result, pytorch_result, "Basic Unfold")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_unfold_with_stride():
    """Test unfold operation with stride > 1."""
    print("\n" + "="*60)
    print("TEST 2: Unfold with Stride=2")
    print("="*60)
    
    # With stride=2, input=(1, 2, 6, 6), kernel=(2,2)
    # Number of blocks = ((6-2)/2 + 1) * ((6-2)/2 + 1) = 3 * 3 = 9
    # Output shape: (1, 8, 9)
    
    input_data = np.arange(72, dtype=np.float32).reshape(1, 2, 6, 6)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_unfold(x):
        return nb.unfold(x, (2, 2), stride=2)
    jitted_nabla_unfold = nb.jit(nabla_unfold)
    nabla_result = jitted_nabla_unfold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.unfold(torch_input, kernel_size=(2, 2), stride=2)
        return compare_with_pytorch(nabla_result, pytorch_result, "Unfold with Stride=2")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_unfold_with_padding():
    """Test unfold operation with padding."""
    print("\n" + "="*60)
    print("TEST 3: Unfold with Padding=1")
    print("="*60)
    
    # With padding=1, input=(1, 2, 4, 4), kernel=(2,2)
    # Padded size = (6, 6)
    # Number of blocks = (6 - 2 + 1) * (6 - 2 + 1) = 5 * 5 = 25
    # Output shape: (1, 8, 25)
    
    input_data = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_unfold(x):
        return nb.unfold(x, (2, 2), padding=1)
    jitted_nabla_unfold = nb.jit(nabla_unfold)
    nabla_result = jitted_nabla_unfold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.unfold(torch_input, kernel_size=(2, 2), padding=1)
        return compare_with_pytorch(nabla_result, pytorch_result, "Unfold with Padding=1")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_unfold_with_dilation():
    """Test unfold operation with dilation."""
    print("\n" + "="*60)
    print("TEST 4: Unfold with Dilation=2")
    print("="*60)
    
    # With dilation=2, input=(1, 2, 6, 6), kernel=(2,2)
    # Effective kernel = (2-1)*2+1 = 3
    # Number of blocks = (6 - 3 + 1) * (6 - 3 + 1) = 4 * 4 = 16
    # Output shape: (1, 8, 16)
    
    input_data = np.arange(72, dtype=np.float32).reshape(1, 2, 6, 6)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_unfold(x):
        return nb.unfold(x, (2, 2), dilation=2)
    jitted_nabla_unfold = nb.jit(nabla_unfold)
    nabla_result = jitted_nabla_unfold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.unfold(torch_input, kernel_size=(2, 2), dilation=2)
        return compare_with_pytorch(nabla_result, pytorch_result, "Unfold with Dilation=2")
    else:
        print("Nabla result:")
        print(nabla_result)
        return True


def test_unfold_combined_params():
    """Test unfold with combined parameters."""
    print("\n" + "="*60)
    print("TEST 5: Unfold with Combined Parameters (stride=2, padding=1, dilation=1)")
    print("="*60)
    
    # Complex case with multiple parameters
    input_data = np.arange(128, dtype=np.float32).reshape(1, 2, 8, 8)
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_unfold(x):
        return nb.unfold(x, (3, 3), stride=2, padding=1, dilation=1)
    jitted_nabla_unfold = nb.jit(nabla_unfold)
    nabla_result = jitted_nabla_unfold(nabla_input)
    
    # PyTorch
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.unfold(torch_input, kernel_size=(3, 3), stride=2, padding=1, dilation=1)
        return compare_with_pytorch(nabla_result, pytorch_result, "Unfold with Combined Parameters")
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
    input_size = (5, 5)
    
    input_data = np.random.randn(batch_size, channels, *input_size).astype(np.float32)
    
    print(f"Input shape: {input_data.shape}")
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_unfold(x):
        return nb.unfold(x, kernel_size)
    jitted_nabla_unfold = nb.jit(nabla_unfold)
    nabla_result = jitted_nabla_unfold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.unfold(torch_input, kernel_size)
        return compare_with_pytorch(nabla_result, pytorch_result, "Larger Batch Size")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_non_square_kernel():
    """Test with non-square kernel."""
    print("\n" + "="*60)
    print("TEST 7: Non-Square Kernel (3x5)")
    print("="*60)
    
    input_data = np.random.randn(1, 2, 8, 10).astype(np.float32)
    kernel_size = (3, 5)
    
    print(f"Input shape: {input_data.shape}")
    
    # Nabla
    nabla_input = nb.tensor(input_data)
    def nabla_unfold(x):
        return nb.unfold(x, kernel_size)
    jitted_nabla_unfold = nb.jit(nabla_unfold)
    nabla_result = jitted_nabla_unfold(nabla_input)
    
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        pytorch_result = F.unfold(torch_input, kernel_size)
        return compare_with_pytorch(nabla_result, pytorch_result, "Non-Square Kernel")
    else:
        print("Nabla result shape:", nabla_result.shape)
        return True


def test_fold_unfold_relationship():
    """Test that fold and unfold are related operations."""
    print("\n" + "="*60)
    print("TEST 8: Fold-Unfold Relationship")
    print("="*60)
    
    # Create input
    input_data = np.random.randn(1, 2, 6, 6).astype(np.float32)
    kernel_size = (2, 2)
    
    # Nabla: unfold then fold
    nabla_input = nb.tensor(input_data)
    
    def nabla_unfold_fold(x):
        unfolded = nb.unfold(x, kernel_size)
        folded = nb.fold(unfolded, (6, 6), kernel_size)
        return folded
    
    jitted_unfold_fold = nb.jit(nabla_unfold_fold)
    nabla_result = jitted_unfold_fold(nabla_input)
    
    # PyTorch: unfold then fold
    if PYTORCH_AVAILABLE:
        torch_input = torch.from_numpy(input_data)
        torch_unfolded = F.unfold(torch_input, kernel_size)
        torch_folded = F.fold(torch_unfolded, (6, 6), kernel_size)
        
        # Calculate divisor
        input_ones = torch.ones_like(torch_input)
        divisor = F.fold(F.unfold(input_ones, kernel_size), (6, 6), kernel_size)
        
        # fold(unfold(input)) == divisor * input (up to some constant)
        print("\nNabla fold(unfold(input)):")
        print(nabla_result.to_numpy())
        print("\nPyTorch fold(unfold(input)):")
        print(torch_folded.numpy())
        print("\nPyTorch divisor:")
        print(divisor.numpy())
        
        # They should match
        return compare_with_pytorch(nabla_result, torch_folded, "Fold-Unfold Relationship")
    else:
        print("Nabla fold(unfold(input)) shape:", nabla_result.shape)
        return True


if __name__ == "__main__":
    if not PYTORCH_AVAILABLE:
        print("\n⚠️  WARNING: PyTorch is not installed. Comparison tests will be skipped.")
        print("To run comparison tests, install PyTorch: pip install torch\n")
    
    results = []
    
    # Original tests
    results.append(("Basic Unfold", test_basic_unfold()))
    results.append(("Unfold with Stride", test_unfold_with_stride()))
    results.append(("Unfold with Padding", test_unfold_with_padding()))
    results.append(("Unfold with Dilation", test_unfold_with_dilation()))
    results.append(("Unfold with Combined Parameters", test_unfold_combined_params()))
    
    # Additional tests
    results.append(("Larger Batch Size", test_larger_batch()))
    results.append(("Non-Square Kernel", test_non_square_kernel()))
    results.append(("Fold-Unfold Relationship", test_fold_unfold_relationship()))
    
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
