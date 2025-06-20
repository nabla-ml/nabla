import numpy as np
from max.driver import Tensor
from max.dtype import DType

print("🔍 FOCUSED BOOLEAN TENSOR CREATION TEST")
print("=" * 60)

# Test the exact MAX library flow for boolean tensor creation

print("Test 1: Boolean array (should work)")
try:
    bool_array_np = np.array([True, False, True], dtype=bool)
    print(
        f"  NumPy array: {bool_array_np}, shape: {bool_array_np.shape}, dtype: {bool_array_np.dtype}"
    )

    # Follow MAX library code path
    is_bool = bool_array_np.dtype == bool
    print(f"  is_bool: {is_bool}")

    if is_bool:
        uint8_view = bool_array_np.view(np.uint8)
        print(
            f"  uint8 view: {uint8_view}, shape: {uint8_view.shape}, dtype: {uint8_view.dtype}"
        )

        tensor = Tensor._from_dlpack(uint8_view)
        print(f"  uint8 tensor: {tensor}")

        # This is where it might fail for scalars
        final_tensor = tensor.view(DType.bool)
        print(f"  ✅ SUCCESS: {final_tensor}")

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\nTest 2: Boolean scalar (the problem case)")
try:
    bool_scalar_np = np.array(True, dtype=bool)
    print(
        f"  NumPy scalar: {bool_scalar_np}, shape: {bool_scalar_np.shape}, dtype: {bool_scalar_np.dtype}"
    )

    # Follow MAX library code path
    is_bool = bool_scalar_np.dtype == bool
    print(f"  is_bool: {is_bool}")

    if is_bool:
        uint8_view = bool_scalar_np.view(np.uint8)
        print(
            f"  uint8 view: {uint8_view}, shape: {uint8_view.shape}, dtype: {uint8_view.dtype}"
        )

        tensor = Tensor._from_dlpack(uint8_view)
        print(f"  uint8 tensor: {tensor}")

        # This is where it fails for scalars
        print("  Attempting tensor.view(DType.bool)...")
        final_tensor = tensor.view(DType.bool)
        print(f"  ✅ SUCCESS: {final_tensor}")

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\nTest 3: Try creating boolean scalar via different path")
try:
    # Create uint8 scalar directly and view as bool
    uint8_scalar = np.array(1, dtype=np.uint8)
    print(f"  uint8 scalar: {uint8_scalar}, shape: {uint8_scalar.shape}")

    tensor = Tensor._from_dlpack(uint8_scalar)
    print(f"  uint8 tensor: {tensor}")

    print("  Attempting tensor.view(DType.bool)...")
    bool_tensor = tensor.view(DType.bool)
    print(f"  ✅ SUCCESS: {bool_tensor}")

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\nTest 4: Investigate tensor properties before view")
try:
    uint8_scalar = np.array(1, dtype=np.uint8)
    tensor = Tensor._from_dlpack(uint8_scalar)

    print(f"  Tensor before view: {tensor}")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Tensor rank: {tensor.rank}")

    # Try to access the problematic property
    print("  Attempting to access shape[-1]...")
    if tensor.shape:
        print(f"  shape[-1]: {tensor.shape[-1]}")
    else:
        print(f"  shape is empty: {tensor.shape}")
        print("  This is the problem! shape[-1] fails on empty tuple")

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback

    traceback.print_exc()
