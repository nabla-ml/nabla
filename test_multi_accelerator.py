#!/usr/bin/env python3
"""Test script demonstrating multi-accelerator handling in move_to_best_device."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nabla.ops.operation import move_to_best_device
from nabla.ops.creation import array
from nabla.utils.max_interop import cpu, accelerator

def test_multi_accelerator_scenarios():
    """Test multi-accelerator device selection scenarios."""
    
    print("Testing multi-accelerator scenarios...")
    
    try:
        # Create multiple accelerator devices
        cpu_device = cpu()
        gpu0 = accelerator(0)
        gpu1 = accelerator(1)  # This might fail if only 1 GPU available
        
        print(f"\nAvailable devices:")
        print(f"  CPU: {cpu_device.label} (is_host: {cpu_device.is_host})")
        print(f"  GPU0: {gpu0.label} (is_host: {gpu0.is_host})")
        print(f"  GPU1: {gpu1.label} (is_host: {gpu1.is_host})")
        
        # Check peer access
        gpu0_can_access_gpu1 = gpu0.can_access(gpu1)
        gpu1_can_access_gpu0 = gpu1.can_access(gpu0)
        print(f"\nPeer access:")
        print(f"  GPU0 -> GPU1: {gpu0_can_access_gpu1}")
        print(f"  GPU1 -> GPU0: {gpu1_can_access_gpu0}")
        
        # Scenario 1: Data spread across multiple accelerators
        print(f"\n--- Scenario 1: Data on both GPUs ---")
        gpu0_arr = array([[1, 2, 3], [4, 5, 6]], device=gpu0)  # 6 elements on GPU0
        gpu1_arr = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device=gpu1)  # 10 elements on GPU1
        
        print(f"Input: GPU0({gpu0_arr.size} elements), GPU1({gpu1_arr.size} elements)")
        
        result = move_to_best_device(gpu0_arr, gpu1_arr)
        result_devices = [arg.device.label for arg in result]
        print(f"Output devices: {result_devices}")
        print(f"Expected: Both on GPU1 (has more data: {gpu1_arr.size} > {gpu0_arr.size})")
        
        # Scenario 2: Peer access consideration
        print(f"\n--- Scenario 2: Peer access affects selection ---")
        # Create scenario where GPU0 has slightly less data but better peer access
        gpu0_big = array([[i]*8 for i in range(8)], device=gpu0)  # 64 elements on GPU0
        gpu1_small = array([[i]*9 for i in range(8)], device=gpu1)  # 72 elements on GPU1
        
        print(f"Input: GPU0({gpu0_big.size} elements), GPU1({gpu1_small.size} elements)")
        print(f"GPU1 has more data ({gpu1_small.size} > {gpu0_big.size})")
        
        result = move_to_best_device(gpu0_big, gpu1_small)
        result_devices = [arg.device.label for arg in result]
        print(f"Output devices: {result_devices}")
        
        if gpu0_can_access_gpu1 and not gpu1_can_access_gpu0:
            print("Expected: Both on GPU0 (peer access bonus despite less data)")
        elif gpu1_can_access_gpu0 and not gpu0_can_access_gpu1:
            print("Expected: Both on GPU1 (more data + peer access)")
        else:
            print("Expected: Both on GPU1 (more data)")
            
    except Exception as e:
        print(f"Multi-GPU test failed (likely only 1 GPU or no GPU available): {e}")
        
        # Fallback: Test single GPU scenario
        print(f"\n--- Fallback: Single GPU scenario ---")
        try:
            cpu_device = cpu()
            gpu_device = accelerator(0)
            
            cpu_arr = array([[i]*20 for i in range(10)], device=cpu_device)  # 200 elements on CPU
            gpu_arr = array([1, 2], device=gpu_device)  # 2 elements on GPU
            
            print(f"Input: CPU({cpu_arr.size} elements), GPU({gpu_arr.size} elements)")
            
            result = move_to_best_device(cpu_arr, gpu_arr)
            result_devices = [arg.device.label for arg in result]
            print(f"Output devices: {result_devices}")
            print("Expected: Both on GPU (accelerator preferred despite less data)")
            
        except Exception as e2:
            print(f"Single GPU test also failed (no GPU available): {e2}")

if __name__ == "__main__":
    test_multi_accelerator_scenarios()
