#!/usr/bin/env python3

import nabla as nb
import numpy as np

def test_simple_grad():
    """Test with the simplest possible case."""
    
    def simple_func(x):
        return nb.sum(x)  # Just sum, no multiplication
    
    x = nb.arange((2, 3))
    print(f"Input shape: {x.shape}")
    
    # Check function output
    result = simple_func(x)
    print(f"Function output shape: {result.shape}")
    print(f"Function output value: {result.to_numpy()}")
    
    # Test gradient
    grad_fn = nb.grad(simple_func)
    
    print("\nGradient XPR:")
    print(nb.xpr(grad_fn, x))
    
    grad = grad_fn(x)
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient value: {grad.to_numpy()}")
    
    # Expected: gradient of sum(x) is all ones with same shape as x
    expected = np.ones_like(x.to_numpy())
    print(f"Expected shape: {expected.shape}")
    print(f"Expected value: {expected}")
    
    print(f"Shapes match: {grad.shape == expected.shape}")
    if grad.shape == expected.shape:
        print(f"Values match: {np.allclose(grad.to_numpy(), expected)}")

if __name__ == "__main__":
    test_simple_grad()
