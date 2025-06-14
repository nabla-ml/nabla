#!/usr/bin/env python3
"""Test script demonstrating automatic device movement in operations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import nabla as nb

def test_automatic_device_movement():
    """Test that operations automatically move arrays to best device."""
    
    print("Testing automatic device movement in operations...")
    
    try:
        # Create arrays on different devices
        cpu_device = nb.cpu()
        gpu_device = nb.accelerator(0)
        
        print(f"\nAvailable devices:")
        print(f"  CPU: {cpu_device.label}")
        print(f"  GPU: {gpu_device.label}")
        
        # Test 1: Binary operation (add)
        print(f"\n--- Test 1: Binary Operation (add) ---")
        cpu_arr = nb.array([[1, 2], [3, 4]], device=cpu_device)  # 4 elements on CPU
        gpu_arr = nb.array([10, 20], device=gpu_device)  # 2 elements on GPU
        
        print(f"Input devices: CPU({cpu_arr.size} elements), GPU({gpu_arr.size} elements)")
        
        # This should automatically move both to GPU (accelerator preferred)
        result = cpu_arr + gpu_arr  # Uses binary addition
        print(f"Result device: {result.device.label}")
        print(f"Expected: GPU (accelerator preferred despite less data)")
        
        # Test 2: Matrix multiplication
        print(f"\n--- Test 2: Matrix Multiplication ---")
        cpu_mat1 = nb.array([[1, 2, 3], [4, 5, 6]], device=cpu_device)  # 6 elements
        gpu_mat2 = nb.array([[1, 2], [3, 4], [5, 6]], device=gpu_device)  # 6 elements
        
        print(f"Input devices: CPU({cpu_mat1.size} elements), GPU({gpu_mat2.size} elements)")
        
        result = cpu_mat1 @ gpu_mat2  # Uses matmul
        print(f"Result device: {result.device.label}")
        print(f"Expected: GPU (accelerator preferred)")
        
        # Test 3: Concatenation
        print(f"\n--- Test 3: Concatenation ---")
        cpu_arr1 = nb.array([[1, 2, 3, 4, 5]] * 10, device=cpu_device)  # 50 elements on CPU
        gpu_arr1 = nb.array([[6, 7]], device=gpu_device)  # 2 elements on GPU
        
        print(f"Input devices: CPU({cpu_arr1.size} elements), GPU({gpu_arr1.size} elements)")
        
        result = nb.concatenate([cpu_arr1, gpu_arr1], axis=0)
        print(f"Result device: {result.device.label}")
        print(f"Expected: GPU (accelerator preferred despite less data)")
        
    except Exception as e:
        print(f"GPU test failed (no GPU available): {e}")
        
        # Fallback: Test CPU-only scenario
        print(f"\n--- Fallback: CPU-only scenario ---")
        cpu_device = nb.cpu()
        
        arr1 = nb.array([[1, 2, 3]] * 5, device=cpu_device)  # 15 elements
        arr2 = nb.array([4, 5], device=cpu_device)  # 2 elements
        
        print(f"Input devices: Both on CPU")
        
        result = arr1[0] + arr2  # Should both stay on CPU
        print(f"Result device: {result.device.label}")
        print(f"Expected: CPU (only available device)")

if __name__ == "__main__":
    test_automatic_device_movement()
