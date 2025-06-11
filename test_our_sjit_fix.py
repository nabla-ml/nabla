#!/usr/bin/env python3
"""Test script to verify our static JIT fix is working correctly."""

import nabla as nb
import numpy as np

def test_basic_sjit():
    """Test basic static JIT functionality."""
    print("=== Testing Basic Static JIT ===")
    
    @nb.sjit
    def simple_func(x, y):
        return x * y
    
    a = nb.array([1., 2., 3.])
    b = nb.array([4., 5., 6.])
    
    result = simple_func(a, b)
    expected = np.array([4., 10., 18.])
    
    assert np.allclose(result.to_numpy(), expected), f"Expected {expected}, got {result.to_numpy()}"
    print("✅ Basic static JIT test passed")

def test_sjit_with_scalars():
    """Test static JIT with scalar arguments (the main fix)."""
    print("=== Testing Static JIT with Scalars ===")
    
    @nb.sjit
    def func_with_scalar(x, factor):
        return x * factor
    
    a = nb.array([1., 2., 3.])
    
    # Test with different scalar values (this was the bug)
    result1 = func_with_scalar(a, 2)
    result2 = func_with_scalar(a, 3)
    result3 = func_with_scalar(a, 4)
    
    expected1 = np.array([2., 4., 6.])
    expected2 = np.array([3., 6., 9.])
    expected3 = np.array([4., 8., 12.])
    
    assert np.allclose(result1.to_numpy(), expected1), f"Expected {expected1}, got {result1.to_numpy()}"
    assert np.allclose(result2.to_numpy(), expected2), f"Expected {expected2}, got {result2.to_numpy()}"
    assert np.allclose(result3.to_numpy(), expected3), f"Expected {expected3}, got {result3.to_numpy()}"
    print("✅ Static JIT with scalars test passed")

def test_rpow_operation():
    """Test the __rpow__ operation we added."""
    print("=== Testing __rpow__ Operation ===")
    
    x = nb.array([2., 3., 4.])
    result = 2.0 ** x  # This should use __rpow__
    expected = np.array([4., 8., 16.])
    
    assert np.allclose(result.to_numpy(), expected), f"Expected {expected}, got {result.to_numpy()}"
    print("✅ __rpow__ test passed")

def test_adam_like_operation():
    """Test an operation similar to what was failing in Adam optimizer."""
    print("=== Testing Adam-like Operation ===")
    
    @nb.sjit
    def adam_like_func(param, step):
        beta1 = 0.9
        bias_correction = 1.0 - beta1 ** step
        return param / bias_correction
    
    param = nb.array([1., 2.])
    
    # Test with different step values
    result1 = adam_like_func(param, 1)
    result2 = adam_like_func(param, 2)
    
    expected1 = np.array([1., 2.]) / (1.0 - 0.9 ** 1)  # = [1., 2.] / 0.1 = [10., 20.]
    expected2 = np.array([1., 2.]) / (1.0 - 0.9 ** 2)  # = [1., 2.] / 0.19 = [5.26, 10.52]
    
    assert np.allclose(result1.to_numpy(), expected1), f"Expected {expected1}, got {result1.to_numpy()}"
    assert np.allclose(result2.to_numpy(), expected2), f"Expected {expected2}, got {result2.to_numpy()}"
    print("✅ Adam-like operation test passed")

if __name__ == "__main__":
    try:
        print("Starting static JIT tests...")
        test_basic_sjit()
        test_sjit_with_scalars()
        test_rpow_operation()
        test_adam_like_operation()
        print("\n🎉 All static JIT tests passed! Our fix is working correctly.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
