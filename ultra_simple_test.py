#!/usr/bin/env python3

import sys
sys.path.append('/Users/tillife/Documents/CodingProjects/nabla')

import nabla as nb
from nabla.core.trafos import vmap

def ultra_simple_test():
    """Ultra simple test to find the root cause."""
    
    # The simplest possible case
    x = nb.array([1.0, 2.0, 3.0])  # shape (3,)
    y = nb.array([4.0, 5.0, 6.0])  # shape (3,)
    
    def simple_add(a, b):
        return a + b
    
    print("=== Ultra simple test ===")
    print(f"Input x shape: {x.shape}")
    print(f"Input y shape: {y.shape}")
    
    # Direct function call
    direct_result = simple_add(x, y)
    print(f"Direct call result shape: {direct_result.shape}")
    print(f"Direct call result: {direct_result.to_numpy()}")
    
    # Vmap with simple inputs
    vmapped_func = vmap(simple_add, in_axes=0)
    vmap_result = vmapped_func(x, y)
    print(f"Vmap result shape: {vmap_result.shape}")
    print(f"Vmap result: {vmap_result.to_numpy()}")
    
    # Test with a function that returns a scalar per batch element
    def scalar_add(a, b):
        result = a + b
        print(f"   Inside function - result shape: {result.shape}")
        return result
    
    print("\n--- With debug prints ---")
    vmapped_func2 = vmap(scalar_add, in_axes=0)
    vmap_result2 = vmapped_func2(x, y)
    print(f"Final vmap result shape: {vmap_result2.shape}")

if __name__ == "__main__":
    ultra_simple_test()
