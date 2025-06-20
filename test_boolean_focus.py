import numpy as np

import nabla as nb

print("🎯 FOCUSED BOOLEAN ARRAY CREATION TEST")
print("=" * 50)

# Test 1: Can we create boolean arrays with specific dtype?
print("\n📋 Test 1: Boolean array creation with dtype specification")
try:
    from max.dtype import DType

    # Try to create boolean array directly
    bool_data = np.array([True, False, True], dtype=bool)
    print(f"NumPy boolean array: {bool_data}, dtype: {bool_data.dtype}")

    # Try to create nabla array from boolean numpy array
    nabla_bool = nb.Array.from_numpy(bool_data)
    print(f"✅ Nabla from numpy boolean array: {nabla_bool}")

except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n📋 Test 2: Boolean scalar creation")
try:
    # Try to create boolean scalar
    bool_scalar = np.array(True, dtype=bool)
    print(
        f"NumPy boolean scalar: {bool_scalar}, dtype: {bool_scalar.dtype}, shape: {bool_scalar.shape}"
    )

    # Try to create nabla array from boolean numpy scalar
    nabla_bool_scalar = nb.Array.from_numpy(bool_scalar)
    print(f"✅ Nabla from numpy boolean scalar: {nabla_bool_scalar}")

except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n📋 Test 3: Manual boolean tensor creation using MAX")
try:
    from max.driver import Tensor

    # Try the same logic as MAX library
    bool_array = np.array([True, False], dtype=bool)
    print(f"Original: {bool_array}, shape: {bool_array.shape}")

    # Convert to uint8 (like MAX does)
    uint8_array = bool_array.view(np.uint8)
    print(f"As uint8: {uint8_array}")

    # Create tensor
    tensor = Tensor.from_numpy(uint8_array)
    print(f"Tensor: {tensor}")

    # Try to view as bool
    bool_tensor = tensor.view(DType.bool)
    print(f"✅ Bool tensor: {bool_tensor}")

except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n📋 Test 4: Manual boolean SCALAR tensor creation using MAX")
try:
    from max.driver import Tensor

    # Try the same logic as MAX library for SCALAR
    bool_scalar = np.array(True, dtype=bool)
    print(f"Original scalar: {bool_scalar}, shape: {bool_scalar.shape}")

    # Convert to uint8 (like MAX does)
    uint8_scalar = bool_scalar.view(np.uint8)
    print(f"As uint8 scalar: {uint8_scalar}, shape: {uint8_scalar.shape}")

    # Create tensor
    tensor = Tensor.from_numpy(uint8_scalar)
    print(f"Tensor: {tensor}")

    # Try to view as bool (THIS IS WHERE IT FAILS)
    bool_tensor = tensor.view(DType.bool)
    print(f"✅ Bool scalar tensor: {bool_tensor}")

except Exception as e:
    print(f"❌ FAILED: {e}")
    print("   This is the exact bug we need to work around!")

print("\n🎯 CONCLUSION")
print("=" * 50)
print("The issue is specifically in MAX's tensor.view(DType.bool) for scalar tensors.")
print("We need a workaround to avoid creating scalar boolean tensors.")
