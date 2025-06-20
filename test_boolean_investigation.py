import numpy as np

import nabla as nb

print("🔍 INVESTIGATING BOOLEAN ARRAY CREATION ISSUES")
print("=" * 60)


def test_boolean_creation_method(description, test_func):
    """Test a specific boolean array creation method"""
    print(f"\n📋 Testing: {description}")
    try:
        result = test_func()
        print(f"   ✅ SUCCESS: {result}")
        print(f"   📊 Shape: {result.shape}, Dtype: {result.dtype}")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {type(e).__name__}: {e}")
        return False


print("\n🧪 BASIC BOOLEAN ARRAY CREATION")
print("-" * 40)

# Test 1: Direct boolean array creation
test_boolean_creation_method("Direct boolean scalar", lambda: nb.array(True))

test_boolean_creation_method(
    "Direct boolean array", lambda: nb.array([True, False, True])
)

# Test 2: NumPy boolean arrays
test_boolean_creation_method(
    "From NumPy boolean scalar", lambda: nb.array(np.bool_(True))
)

test_boolean_creation_method(
    "From NumPy boolean array", lambda: nb.array(np.array([True, False, True]))
)

print("\n🔢 COMPARISON-BASED BOOLEAN CREATION")
print("-" * 40)

# Test 3: Using comparison operations (the problematic ones)
test_boolean_creation_method(
    "nb.equal scalar results", lambda: nb.equal(nb.array(1), nb.array(1))
)

test_boolean_creation_method(
    "nb.equal array results", lambda: nb.equal(nb.array([1, 2, 3]), nb.array([1, 0, 3]))
)

test_boolean_creation_method(
    "Modulo + equal scalar", lambda: nb.equal(nb.array(4) % nb.array(2), nb.array(0))
)

test_boolean_creation_method(
    "Modulo + equal array", lambda: nb.equal(nb.arange((4,)) % nb.array(2), nb.array(0))
)

print("\n🎯 LOGICAL_NOT OPERATION TESTS")
print("-" * 40)

# Test 4: Direct logical_not tests
test_boolean_creation_method(
    "logical_not on direct boolean scalar", lambda: nb.logical_not(nb.array(True))
)

test_boolean_creation_method(
    "logical_not on direct boolean array",
    lambda: nb.logical_not(nb.array([True, False, True])),
)


# Test 5: The problematic combination
def test_full_pipeline_scalar():
    """Test the full pipeline that's failing"""
    print("\n🔧 FULL PIPELINE TEST (SCALAR)")
    print("-" * 30)

    # Step by step
    print("Step 1: Create scalar with arange")
    scalar_arr = nb.arange(())  # This should create a scalar 0
    print(f"   Result: {scalar_arr}")

    print("Step 2: Apply modulo")
    mod_result = scalar_arr % nb.array(2)
    print(f"   Result: {mod_result}")

    print("Step 3: Compare with equal")
    bool_result = nb.equal(mod_result, nb.array(0))
    print(f"   Result: {bool_result}")

    print("Step 4: Apply logical_not")
    final_result = nb.logical_not(bool_result)
    print(f"   Result: {final_result}")

    return final_result


def test_full_pipeline_array():
    """Test the full pipeline with arrays"""
    print("\n🔧 FULL PIPELINE TEST (ARRAY)")
    print("-" * 30)

    # Step by step
    print("Step 1: Create array with arange")
    arr = nb.arange((4,))  # [0, 1, 2, 3]
    print(f"   Result: {arr}")

    print("Step 2: Apply modulo")
    mod_result = arr % nb.array(2)
    print(f"   Result: {mod_result}")

    print("Step 3: Compare with equal")
    bool_result = nb.equal(mod_result, nb.array(0))
    print(f"   Result: {bool_result}")

    print("Step 4: Apply logical_not")
    final_result = nb.logical_not(bool_result)
    print(f"   Result: {final_result}")

    return final_result


# Run the pipeline tests
try:
    test_full_pipeline_scalar()
except Exception as e:
    print(f"❌ SCALAR PIPELINE FAILED: {type(e).__name__}: {e}")

try:
    test_full_pipeline_array()
except Exception as e:
    print(f"❌ ARRAY PIPELINE FAILED: {type(e).__name__}: {e}")

print("\n📊 SUMMARY")
print("=" * 60)
print("This test helps us identify exactly where boolean scalar handling breaks")
print("and what operations work vs. fail with boolean scalars.")
