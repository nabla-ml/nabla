import numpy as np
from max.driver import Tensor

print("🔍 TESTING MAX TENSOR BOOLEAN SCALAR CREATION")
print("=" * 60)

# Test 1: Create float scalar
print("Test 1: Float scalar")
try:
    float_np = np.array(1.0)
    float_tensor = Tensor.from_numpy(float_np)
    print(f"✅ SUCCESS: {float_tensor}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: Create boolean array
print("\nTest 2: Boolean array")
try:
    bool_np = np.array([True, False])
    bool_tensor = Tensor.from_numpy(bool_np)
    print(f"✅ SUCCESS: {bool_tensor}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Create boolean scalar (the problematic case)
print("\nTest 3: Boolean scalar - direct")
try:
    bool_scalar_np = np.array(True)
    print(
        f"NumPy array: {bool_scalar_np}, shape: {bool_scalar_np.shape}, dtype: {bool_scalar_np.dtype}"
    )
    bool_scalar_tensor = Tensor.from_numpy(bool_scalar_np)
    print(f"✅ SUCCESS: {bool_scalar_tensor}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Create boolean scalar via numpy comparison
print("\nTest 4: Boolean scalar via numpy comparison")
try:
    comparison_result = np.array(1) == np.array(1)
    print(
        f"Comparison result: {comparison_result}, shape: {comparison_result.shape}, dtype: {comparison_result.dtype}"
    )
    comparison_tensor = Tensor.from_numpy(comparison_result)
    print(f"✅ SUCCESS: {comparison_tensor}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Try workaround - create via reshape
print("\nTest 5: Workaround - create 1D then reshape")
try:
    bool_1d = np.array([True])
    tensor_1d = Tensor.from_numpy(bool_1d)
    print(f"1D tensor: {tensor_1d}")
    # Try to reshape to scalar using proper method
    from max.dtype import DType

    scalar_tensor = tensor_1d.view(DType.bool, shape=())
    print(f"✅ SUCCESS: {scalar_tensor}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback

    traceback.print_exc()
