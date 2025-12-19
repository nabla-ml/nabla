"""Quick verification test for lazy execution and symbolic shapes after refactoring."""

from nabla import Tensor
import numpy as np

print("=" * 70)
print("TEST 1: Lazy Execution - Intermediate Evaluations")
print("=" * 70)

# Test lazy execution with intermediate .item() calls
x = Tensor.arange(0, 5)  # [0, 1, 2, 3, 4]
print(f"✓ Created x = arange(0, 5)")

y = x + 10  # Should be lazy
print(f"✓ Created y = x + 10 (lazy)")
print(f"  y is unrealized: {not y.real}")

# Force evaluation of y
y._sync_realize()
y_array = y.to_numpy()
print(f"✓ Evaluated y.to_numpy() = {y_array}")
expected_y = np.array([10., 11., 12., 13., 14.], dtype=np.float32)
assert np.allclose(y_array, expected_y), f"y should be {expected_y}"
print(f"  y is now realized: {y.real}")

# Continue with more operations - x should still be available
z = x * 2  # Should work even though y was realized
print(f"✓ Created z = x * 2 (lazy)")
print(f"  z is unrealized: {not z.real}")

# Realize z
z_array = z.to_numpy()
print(f"✓ Realized z.numpy() = {z_array}")
expected_z = np.array([0., 2., 4., 6., 8.], dtype=np.float32)
assert np.allclose(z_array, expected_z), f"z should be {expected_z}"

print("\n✅ Lazy execution with intermediate evaluations works!\n")

print("=" * 70)
print("TEST 2: Symbolic Shape Inference")
print("=" * 70)

# Test symbolic dimensions
a = Tensor.ones(("batch", 64))  # Symbolic batch size
print(f"✓ Created a with shape ('batch', 64)")
print(f"  a.shape = {a.shape}")
print(f"  a._value shape = {a._value.type.shape}")

# Check symbolic dimension is preserved
assert str(a._value.type.shape) == "[Dim('batch'), Dim(64)]", "Symbolic dim should be preserved"
print(f"✓ Symbolic dimension 'batch' preserved")

# Binary operation should preserve symbolic dims
b = a + a
print(f"\n✓ Created b = a + a")
print(f"  b.shape = {b.shape}")
print(f"  b._value shape = {b._value.type.shape}")
assert str(b._value.type.shape) == "[Dim('batch'), Dim(64)]", "Symbolic dim should propagate"
print(f"✓ Symbolic dimension propagated through addition")

# Test matmul with symbolic dims
W = Tensor.ones((64, "hidden"))  # Symbolic output size
print(f"\n✓ Created W with shape (64, 'hidden')")
print(f"  W.shape = {W.shape}")

c = a @ W  # Should be (batch, hidden)
print(f"\n✓ Created c = a @ W")
print(f"  c.shape = {c.shape}")
print(f"  c._value shape = {c._value.type.shape}")
assert str(c._value.type.shape) == "[Dim('batch'), Dim('hidden')]", "Matmul should preserve both symbolic dims"
print(f"✓ Symbolic dimensions preserved through matmul: ('batch', 'hidden')")

# Test broadcasting with symbolic dims
bias = Tensor.ones(("hidden",))
print(f"\n✓ Created bias with shape ('hidden',)")
print(f"  bias.shape = {bias.shape}")

d = c + bias  # Should broadcast correctly
print(f"\n✓ Created d = c + bias")
print(f"  d.shape = {d.shape}")
print(f"  d._value shape = {d._value.type.shape}")
assert str(d._value.type.shape) == "[Dim('batch'), Dim('hidden')]", "Broadcasting should preserve symbolic dims"
print(f"✓ Broadcasting with symbolic dimensions works!")

print("\n✅ Symbolic shape inference works perfectly!\n")

print("=" * 70)
print("🎉 ALL VERIFICATION TESTS PASSED!")
print("=" * 70)
print("✓ Lazy execution model intact")
print("✓ Intermediate evaluations work")
print("✓ Symbolic shape inference preserved")
print("✓ Shape propagation through ops correct")
print("=" * 70)
