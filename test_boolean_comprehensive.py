import nabla as nb

print("🧪 COMPREHENSIVE BOOLEAN SCALAR OPERATIONS TEST")
print("=" * 60)

# Test all boolean-producing operations with scalars
test_cases = [
    ("nb.equal(1, 1)", lambda: nb.equal(nb.array(1), nb.array(1))),
    ("nb.equal(1, 2)", lambda: nb.equal(nb.array(1), nb.array(2))),
    ("nb.not_equal(1, 1)", lambda: nb.not_equal(nb.array(1), nb.array(1))),
    ("nb.not_equal(1, 2)", lambda: nb.not_equal(nb.array(1), nb.array(2))),
    ("nb.greater_equal(2, 1)", lambda: nb.greater_equal(nb.array(2), nb.array(1))),
    ("nb.greater_equal(1, 2)", lambda: nb.greater_equal(nb.array(1), nb.array(2))),
    ("nb.logical_not(True)", lambda: nb.logical_not(nb.array(True))),
    ("nb.logical_not(False)", lambda: nb.logical_not(nb.array(False))),
]

all_passed = True

for desc, test_func in test_cases:
    try:
        result = test_func()
        print(f"✅ {desc:<25} → {result}")
    except Exception as e:
        print(f"❌ {desc:<25} → FAILED: {e}")
        all_passed = False

print(f"\n📊 RESULT: {'All tests passed!' if all_passed else 'Some tests failed.'}")

# Test the modulo operation for boolean generation
print("\n🔢 BOOLEAN DATA GENERATION TEST")
print("-" * 40)

try:
    # This was the original failing use case
    arange_data = nb.arange((4,))
    mod_data = arange_data % 2
    bool_data = nb.equal(mod_data, nb.array(0))
    logical_not_data = nb.logical_not(bool_data)

    print(f"arange((4,)): {arange_data}")
    print(f"arange % 2:   {mod_data}")
    print(f"equal(%, 0):  {bool_data}")
    print(f"logical_not:  {logical_not_data}")
    print("✅ Boolean data generation pipeline works!")

except Exception as e:
    print(f"❌ Boolean data generation failed: {e}")
    import traceback

    traceback.print_exc()
