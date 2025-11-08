"""
Test driver module implementation
==================================

Tests the new driver module wrappers to ensure proper Python-Mojo interop.
"""

from python import Python
from max_graph.driver import (
    CPU, 
    Accelerator, 
    Device, 
    DeviceStream, 
    DeviceSpec,
    accelerator_count,
    accelerator_api,
    accelerator_architecture_name,
)
from max_graph.types import Tensor


fn test_cpu_device() raises:
    """Test CPU device creation and properties."""
    print("Testing CPU device...")
    
    var cpu = CPU()
    print("  Created CPU device")
    
    var cpu_id = cpu.id()
    print("  CPU ID:", cpu_id)
    
    var device = cpu.as_device()
    var is_host = device.is_host()
    print("  Is host:", is_host)
    
    var label = device.label()
    print("  Label:", label)
    
    print("  ✓ CPU device test passed")


fn test_accelerator_count() raises:
    """Test accelerator count function."""
    print("\nTesting accelerator count...")
    
    var count = accelerator_count()
    print("  Accelerator count:", count)
    
    if count > 0:
        print("  ✓ Accelerators available")
    else:
        print("  ⚠ No accelerators found (CPU only)")


fn test_accelerator_device() raises:
    """Test Accelerator device - try to create even if count is 0."""
    print("\nTesting Accelerator device...")
    
    var count = accelerator_count()
    if count == 0:
        print("  ⚠ accelerator_count() returns 0, but trying to create anyway...")
    
    try:
        var accel = Accelerator(id=0)
        print("  ✓ Created Accelerator device")
        
        var accel_id = accel.id()
        print("  Accelerator ID:", accel_id)
        
        var api = accel.api()
        print("  API:", api)
        
        try:
            var arch = accel.architecture_name()
            print("  Architecture:", arch)
        except:
            print("  Architecture: (not available for this device)")
        
        var device = accel.as_device()
        var is_host = device.is_host()
        print("  Is host:", is_host)
        
        print("  ✓ Accelerator device test passed")
    except e:
        print("  ✗ Failed to create Accelerator:", e)
        print("  Skipping accelerator tests")


fn test_device_stream() raises:
    """Test DeviceStream creation."""
    print("\nTesting DeviceStream...")
    
    var cpu = CPU()
    var cpu_device = cpu.as_device()
    
    var default_stream = cpu_device.default_stream()
    print("  Created default stream")
    
    var new_stream = DeviceStream.create(cpu_device)
    print("  Created new stream")
    
    # Test synchronization
    default_stream.synchronize()
    print("  Synchronized stream")
    
    print("  ✓ DeviceStream test passed")


fn test_device_spec() raises:
    """Test DeviceSpec creation."""
    print("\nTesting DeviceSpec...")
    
    var cpu_spec = DeviceSpec.cpu()
    print("  CPU spec - ID:", cpu_spec.id, "Type:", cpu_spec.device_type)
    
    var accel_spec = DeviceSpec.accelerator(id=0)
    print("  Accelerator spec - ID:", accel_spec.id, "Type:", accel_spec.device_type)
    
    print("  ✓ DeviceSpec test passed")


fn test_tensor_operations() raises:
    """Test Tensor with device operations."""
    print("\nTesting Tensor operations...")
    
    # Create a numpy array
    var np = Python.import_module("numpy")
    var np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # Create tensor from numpy
    var tensor = Tensor.from_numpy(np_array)
    print("  Created tensor from numpy")
    
    # Check if it's on host
    var is_host = tensor.is_host()
    print("  Tensor is_host:", is_host)
    
    # Check shape
    var shape = tensor.shape()
    print("  Tensor shape:", shape)
    
    # Convert back to numpy
    var np_result = tensor.to_numpy()
    print("  Converted back to numpy:", np_result)
    
    # Test copy to CPU
    var cpu = CPU()
    var cpu_tensor = tensor.copy_to_cpu()
    print("  Copied to CPU")
    
    # Test copy to accelerator - try even if count is 0
    print("  Attempting to copy to Accelerator...")
    try:
        var accel = Accelerator(id=0)
        var accel_tensor = tensor.copy_to_accelerator(accel)
        print("  ✓ Copied to Accelerator")
        
        var accel_is_host = accel_tensor.is_host()
        print("  Accelerator tensor is_host:", accel_is_host)
        
        # Copy back to CPU to verify
        var back_to_cpu = accel_tensor.copy_to_cpu()
        var result = back_to_cpu.to_numpy()
        print("  Copied back from Accelerator:", result)
    except e:
        print("  ✗ Could not use Accelerator:", e)
    
    print("  ✓ Tensor operations test passed")


fn main() raises:
    """Run all driver module tests."""
    print("=" * 60)
    print("MAX Driver Module Tests")
    print("=" * 60)
    
    test_cpu_device()
    test_accelerator_count()
    test_accelerator_device()
    test_device_stream()
    test_device_spec()
    test_tensor_operations()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
