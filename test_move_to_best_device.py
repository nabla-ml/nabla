#!/usr/bin/env python3
"""Test script for move_to_best_device function."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nabla.ops.operation import move_to_best_device
from nabla.ops.creation import array
from nabla.utils.max_interop import cpu, accelerator

def test_move_to_best_device():
    """Test the move_to_best_device function with different scenarios."""
    
    print("Testing move_to_best_device function...")
    
    # Test 1: All arrays on CPU - should move to device with most data
    print("\n--- Test 1: All CPU arrays ---")
    cpu_device = cpu()
    arr1 = array([1, 2, 3, 4], device=cpu_device)  # 4 elements
    arr2 = array([[1, 2], [3, 4], [5, 6]], device=cpu_device)  # 6 elements
    arr3 = array([1, 2], device=cpu_device)  # 2 elements
    
    result = move_to_best_device(arr1, arr2, arr3)
    print(f"Input devices: {[arg.device.label for arg in [arr1, arr2, arr3]]}")
    print(f"Output devices: {[arg.device.label for arg in result]}")
    print(f"Array sizes: {[arg.size for arg in [arr1, arr2, arr3]]}")
    print("Expected: All should be on same device (CPU with most data)")
    
    # Test 2: Mixed CPU and GPU - should prefer GPU even with less data
    print("\n--- Test 2: Mixed CPU and GPU ---")
    try:
        gpu_device = accelerator(0)  # This might fail if no GPU available
        
        cpu_arr = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device=cpu_device)  # 10 elements on CPU
        gpu_arr = array([1, 2], device=gpu_device)  # 2 elements on GPU
        
        result = move_to_best_device(cpu_arr, gpu_arr)
        print(f"Input devices: CPU({cpu_arr.size} elements), GPU({gpu_arr.size} elements)")
        print(f"Output devices: {[arg.device.label for arg in result]}")
        print("Expected: Both should be on GPU (accelerator preferred)")
        
    except Exception as e:
        print(f"GPU test skipped (no GPU available): {e}")
    
    # Test 3: Scalar conversion
    print("\n--- Test 3: Scalar conversion ---")
    arr = array([1, 2, 3, 4])
    scalar = 5.0
    
    result = move_to_best_device(arr, scalar)
    print(f"Input types: Array({arr.size} elements), scalar")
    print(f"Output types: {[type(arg).__name__ for arg in result]}")
    print(f"Output devices: {[arg.device.label for arg in result]}")
    print("Expected: Both should be Arrays on same device")

if __name__ == "__main__":
    test_move_to_best_device()
