#!/usr/bin/env python3
"""
Test to verify that step slicing properly raises NotImplementedError for VJP
"""
import nabla as nb
import numpy as np

def test_step_limitation():
    """Test that step slicing raises NotImplementedError for VJP"""
    print("Testing step slicing limitation...")
    
    # Create test input
    x = nb.arange((10,))
    
    # Define a function that uses step slicing
    def slice_with_step(x):
        return x[::2]  # Every 2nd element
    
    # Forward pass should work fine
    result = slice_with_step(x)
    print(f"Forward pass works: input shape {x.shape} -> output shape {result.shape}")
    print(f"Result: {result.to_numpy()}")
    
    # VJP should raise NotImplementedError
    try:
        cotangent = nb.ones(result.shape)
        primals_out, vjp_fun = nb.vjp(slice_with_step, x)
        vjp_result = vjp_fun(cotangent)
        print("❌ ERROR: VJP should have raised NotImplementedError!")
        return False
    except NotImplementedError as e:
        print(f"✅ VJP correctly raised NotImplementedError: {e}")
        return True
    except Exception as e:
        print(f"❌ VJP raised unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_step_limitation()
    if success:
        print("\n🎉 Step slicing limitation test passed!")
    else:
        print("\n💥 Step slicing limitation test failed!")
