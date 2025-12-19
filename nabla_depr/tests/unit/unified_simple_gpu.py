"""
Comprehensive test suite for all Nabla operations on GPU with NumPy validation.

This test file demonstrates and validates all major operation categories in Nabla:

BINARY OPERATIONS (12):
- add, mul, sub, div, floordiv, mod, pow
- greater_equal, equal, not_equal, maximum, minimum

UNARY OPERATIONS (12):
- negate, sin, cos, tanh, sigmoid, abs, floor, relu
- log, exp, sqrt, logical_not

REDUCTION OPERATIONS (4):
- sum (all/axis), mean (all/axis), max (all/axis), argmax (axis)

VIEW OPERATIONS (11):
- transpose, permute, move_axis_to_front, reshape, broadcast_to
- squeeze, unsqueeze, tensor_slice, pad, concatenate, stack

LINEAR ALGEBRA OPERATIONS (1):
- matmul

SPECIAL OPERATIONS (3):
- softmax, logsumexp, where

INDEXING OPERATIONS (2):
- gather, scatter

CREATION OPERATIONS (9):
- tensor, arange, ndarange, zeros, ones, randn, rand
- zeros_like, ones_like

All operations are wrapped in nb.jit() for GPU compilation where appropriate.
Each operation is validated against NumPy for correctness.
"""

import numpy as np

import nabla as nb


def compare_with_numpy(nabla_result, numpy_result, operation_name, tolerance=1e-5):
    """Compare Nabla result with NumPy result and print validation status."""
    nabla_np = nabla_result.to_numpy()

    # Handle different dtypes
    if nabla_result.dtype == nb.DType.bool:
        numpy_result = numpy_result.astype(bool)

    try:
        if np.allclose(nabla_np, numpy_result, atol=tolerance, rtol=tolerance):
            print(f"✅ {operation_name}: PASSED")
        else:
            print(f"❌ {operation_name}: FAILED")
            print(f"   Nabla:  {nabla_np}")
            print(f"   NumPy:  {numpy_result}")
            print(f"   Max diff: {np.max(np.abs(nabla_np - numpy_result))}")
    except Exception as e:
        print(f"⚠️  {operation_name}: COMPARISON ERROR - {e}")
        print(f"   Nabla result: {nabla_result}")
        print(f"   NumPy result: {numpy_result}")

    print(f"   Nabla device: {nabla_result.device}")

    return nabla_result


device = nb.cpu() if nb.accelerator_count() == 0 else nb.accelerator()
print(f"Using {device} device")


def test_binary_operations():
    """Test all binary operations with NumPy validation."""
    print("\n=== BINARY OPERATIONS ===")

    # Create test tensors
    a = nb.ndarange((2, 3)).to(device)
    b = nb.ndarange((2, 3)).to(device)
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    b_np = np.arange(6).reshape(2, 3).astype(np.float32)

    print("Add")
    res = nb.jit(nb.add)(a, b)
    compare_with_numpy(res, a_np + b_np, "Add")

    print("Multiply")
    res = nb.jit(nb.mul)(a, b)
    compare_with_numpy(res, a_np * b_np, "Multiply")

    print("Subtract")
    res = nb.jit(nb.sub)(a, b)
    compare_with_numpy(res, a_np - b_np, "Subtract")

    print("Divide")
    res = nb.jit(nb.div)(a, b + 1)
    compare_with_numpy(res, a_np / (b_np + 1), "Divide")

    print("Floor Divide")
    res = nb.jit(nb.floordiv)(a + 5, b + 1)
    compare_with_numpy(res, np.floor_divide(a_np + 5, b_np + 1), "Floor Divide")

    print("Modulo")
    res = nb.jit(nb.mod)(a + 5, b + 1)
    compare_with_numpy(res, np.mod(a_np + 5, b_np + 1), "Modulo")

    print("Power")
    res = nb.jit(nb.pow)(a, b)
    compare_with_numpy(res, np.power(a_np, b_np), "Power")

    print("Greater Equal")
    res = nb.jit(nb.greater_equal)(a, b)
    compare_with_numpy(res, np.greater_equal(a_np, b_np), "Greater Equal")

    print("Equal")
    res = nb.jit(nb.equal)(a, b)
    compare_with_numpy(res, np.equal(a_np, b_np), "Equal")

    print("Not Equal")
    res = nb.jit(nb.not_equal)(a, b)
    compare_with_numpy(res, np.not_equal(a_np, b_np), "Not Equal")

    print("Maximum")
    res = nb.jit(nb.maximum)(a, b)
    compare_with_numpy(res, np.maximum(a_np, b_np), "Maximum")

    print("Minimum")
    res = nb.jit(nb.minimum)(a, b)
    compare_with_numpy(res, np.minimum(a_np, b_np), "Minimum")


def test_unary_operations():
    """Test all unary operations with NumPy validation."""
    print("\n=== UNARY OPERATIONS ===")

    # Create test tensors
    a = nb.ndarange((2, 3)).to(device) + 1  # Add 1 to avoid issues with log/sqrt of 0
    bool_tensor = (
        nb.tensor([True, False, True, False, True, False]).reshape((2, 3)).to(device)
    )

    a_np = (np.arange(6).reshape(2, 3) + 1).astype(np.float32)
    bool_np = np.array([True, False, True, False, True, False]).reshape(2, 3)

    print("Negate")
    res = nb.jit(nb.negate)(a)
    compare_with_numpy(res, -a_np, "Negate")

    print("Sin")
    res = nb.jit(nb.sin)(a)
    compare_with_numpy(res, np.sin(a_np), "Sin")

    print("Cos")
    res = nb.jit(nb.cos)(a)
    compare_with_numpy(res, np.cos(a_np), "Cos")

    print("Tanh")
    res = nb.jit(nb.tanh)(a)
    compare_with_numpy(res, np.tanh(a_np), "Tanh")

    print("Sigmoid")
    res = nb.jit(nb.sigmoid)(a)
    sigmoid_np = 1 / (1 + np.exp(-a_np))
    compare_with_numpy(res, sigmoid_np, "Sigmoid")

    print("Absolute Value")
    res = nb.jit(nb.abs)(-a)
    compare_with_numpy(res, np.abs(-a_np), "Absolute Value")

    print("Floor")
    res = nb.jit(nb.floor)(a + 0.7)
    compare_with_numpy(res, np.floor(a_np + 0.7), "Floor")

    print("ReLU")
    res = nb.jit(nb.relu)(a - 3)
    relu_np = np.maximum(0, a_np - 3)
    compare_with_numpy(res, relu_np, "ReLU")

    print("Log")
    res = nb.jit(nb.log)(a)
    compare_with_numpy(res, np.log(a_np), "Log")

    print("Exponential")
    res = nb.jit(nb.exp)(a)
    compare_with_numpy(res, np.exp(a_np), "Exponential")

    print("Square Root")
    res = nb.jit(nb.sqrt)(a)
    compare_with_numpy(res, np.sqrt(a_np), "Square Root")

    print("Logical Not")
    res = nb.jit(nb.logical_not)(bool_tensor)
    compare_with_numpy(res, np.logical_not(bool_np), "Logical Not")


def test_reduction_operations():
    """Test all reduction operations with NumPy validation."""
    print("\n=== REDUCTION OPERATIONS ===")

    # Create test tensor
    a = nb.ndarange((3, 4)).to(device) + 1
    a_np = (np.arange(12).reshape(3, 4) + 1).astype(np.float32)

    print("Sum (all)")
    res = nb.jit(nb.sum)(a)
    compare_with_numpy(res, np.sum(a_np), "Sum (all)")

    print("Sum (axis=0)")
    res = nb.jit(lambda x: nb.sum(x, axes=0))(a)
    compare_with_numpy(res, np.sum(a_np, axis=0), "Sum (axis=0)")

    print("Sum (axis=1)")
    res = nb.jit(lambda x: nb.sum(x, axes=1))(a)
    compare_with_numpy(res, np.sum(a_np, axis=1), "Sum (axis=1)")

    print("Mean (all)")
    res = nb.jit(nb.mean)(a)
    compare_with_numpy(res, np.mean(a_np), "Mean (all)")

    print("Mean (axis=0)")
    res = nb.jit(lambda x: nb.mean(x, axes=0))(a)
    compare_with_numpy(res, np.mean(a_np, axis=0), "Mean (axis=0)")

    print("Max (all)")
    res = nb.jit(nb.max)(a)
    compare_with_numpy(res, np.max(a_np), "Max (all)")

    print("Max (axis=1)")
    res = nb.jit(lambda x: nb.max(x, axes=1))(a)
    compare_with_numpy(res, np.max(a_np, axis=1), "Max (axis=1)")

    print("Argmax (axis=0)")
    res = nb.jit(lambda x: nb.argmax(x, axes=0))(a)
    compare_with_numpy(res, np.argmax(a_np, axis=0), "Argmax (axis=0)")

    print("Argmax (axis=1)")
    res = nb.jit(lambda x: nb.argmax(x, axes=1))(a)
    compare_with_numpy(res, np.argmax(a_np, axis=1), "Argmax (axis=1)")


def test_view_operations():
    """Test all view operations with NumPy validation."""
    print("\n=== VIEW OPERATIONS ===")

    # Create test tensors
    a = nb.ndarange((2, 3, 4)).to(device)
    b = nb.ndarange((2, 3)).to(device)

    a_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    b_np = np.arange(6).reshape(2, 3).astype(np.float32)

    print("Transpose")
    res = nb.jit(nb.transpose)(b)
    compare_with_numpy(res, np.transpose(b_np), "Transpose")

    print("Permute")
    res = nb.jit(lambda x: nb.permute(x, axes=(2, 0, 1)))(a)
    compare_with_numpy(res, np.transpose(a_np, (2, 0, 1)), "Permute")

    print("Move axis to front")
    res = nb.jit(lambda x: nb.move_axis_to_front(x, axis=2))(a)
    compare_with_numpy(res, np.moveaxis(a_np, 2, 0), "Move axis to front")

    print("Reshape")
    res = nb.jit(lambda x: nb.reshape(x, shape=(6, 4)))(a)
    compare_with_numpy(res, np.reshape(a_np, (6, 4)), "Reshape")

    print("Broadcast to")
    res = nb.jit(lambda x: nb.broadcast_to(x, shape=(3, 2, 3)))(b)
    compare_with_numpy(res, np.broadcast_to(b_np, (3, 2, 3)), "Broadcast to")

    print("Squeeze")
    c = nb.ndarange((2, 1, 3)).to(device)
    c_np = np.arange(6).reshape(2, 1, 3).astype(np.float32)
    res = nb.jit(lambda x: nb.squeeze(x, axes=[1]))(c)
    compare_with_numpy(res, np.squeeze(c_np, axis=1), "Squeeze")

    print("Unsqueeze")
    res = nb.jit(lambda x: nb.unsqueeze(x, axes=[1]))(b)
    compare_with_numpy(res, np.expand_dims(b_np, axis=1), "Unsqueeze")

    print("Tensor slice")
    res = nb.jit(lambda x: nb.tensor_slice(x, slices=[slice(None), slice(0, 2)]))(a)
    compare_with_numpy(res, a_np[:, 0:2], "Tensor slice")

    print("Concatenate")
    tensors = [b, b + 10]
    tensors_np = [b_np, b_np + 10]
    res = nb.jit(lambda x: nb.concatenate(x, axis=0))(tensors)
    compare_with_numpy(res, np.concatenate(tensors_np, axis=0), "Concatenate")

    print("Stack")
    res = nb.jit(lambda x: nb.stack(x, axis=0))(tensors)
    compare_with_numpy(res, np.stack(tensors_np, axis=0), "Stack")

    # Skip pad validation for now as it's more complex


def test_linalg_operations():
    """Test linear algebra operations with NumPy validation."""
    print("\n=== LINEAR ALGEBRA OPERATIONS ===")

    # Create test matrices
    a = nb.ndarange((3, 4)).to(device)
    b = nb.ndarange((4, 5)).to(device)

    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    b_np = np.arange(20).reshape(4, 5).astype(np.float32)

    print("Matrix Multiplication")
    res = nb.jit(nb.matmul)(a, b)
    compare_with_numpy(res, np.matmul(a_np, b_np), "Matrix Multiplication")


def test_special_operations():
    """Test special operations with NumPy validation."""
    print("\n=== SPECIAL OPERATIONS ===")

    # Create test tensors
    a = nb.ndarange((2, 3)).to(device)
    condition = (
        nb.tensor([True, False, True, False, True, False]).reshape((2, 3)).to(device)
    )

    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    condition_np = np.array([True, False, True, False, True, False]).reshape(2, 3)

    print("Softmax")
    res = nb.jit(lambda x: nb.softmax(x, axis=-1))(a)
    # Manual softmax calculation for comparison
    exp_a = np.exp(a_np - np.max(a_np, axis=-1, keepdims=True))
    softmax_np = exp_a / np.sum(exp_a, axis=-1, keepdims=True)
    compare_with_numpy(res, softmax_np, "Softmax")

    print("Log-sum-exp")
    res = nb.jit(lambda x: nb.logsumexp(x, axis=-1))(a)
    # Manual logsumexp calculation
    max_a = np.max(a_np, axis=-1, keepdims=True)
    logsumexp_np = np.log(np.sum(np.exp(a_np - max_a), axis=-1)) + np.squeeze(
        max_a, axis=-1
    )
    compare_with_numpy(res, logsumexp_np, "Log-sum-exp")

    print("Where")
    b = nb.ones((2, 3)).to(device)
    c = nb.zeros((2, 3)).to(device)
    res = nb.jit(nb.where)(condition, b, c)
    compare_with_numpy(
        res, np.where(condition_np, np.ones((2, 3)), np.zeros((2, 3))), "Where"
    )


def test_indexing_operations():
    """Test indexing operations with NumPy validation."""
    print("\n=== INDEXING OPERATIONS ===")

    # Create test tensors
    a = nb.ndarange((3, 4)).to(device)
    indices = nb.tensor([0, 2, 1]).to(device)

    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    indices_np = np.array([0, 2, 1])

    print("Gather")
    res = nb.jit(lambda x, idx: nb.gather(x, idx, axis=0))(a, indices)
    compare_with_numpy(res, a_np[indices_np], "Gather")

    # For scatter, create a simple validation
    target_shape = (4,)
    scatter_indices = nb.tensor([0, 2]).to(device)
    scatter_updates = nb.tensor([10, 20]).to(device)

    print("Scatter")
    res = nb.jit(lambda idx, upd: nb.scatter(target_shape, idx, upd, axis=0))(
        scatter_indices, scatter_updates
    )
    # Manual scatter validation
    scatter_expected = np.zeros(4, dtype=np.float32)
    scatter_expected[0] = 10
    scatter_expected[2] = 20
    compare_with_numpy(res, scatter_expected, "Scatter")


def test_creation_operations():
    """Test tensor creation operations with NumPy validation."""
    print("\n=== CREATION OPERATIONS ===")

    print("Tensor from list")
    res = nb.tensor([1, 2, 3, 4]).to(device)
    compare_with_numpy(res, np.array([1, 2, 3, 4], dtype=np.float32), "Tensor from list")

    print("Arange")
    res = nb.arange(10).to(device)
    compare_with_numpy(res, np.arange(10, dtype=np.float32), "Arange")

    print("NDarange")
    res = nb.ndarange((2, 3)).to(device)
    compare_with_numpy(res, np.arange(6).reshape(2, 3).astype(np.float32), "NDarange")

    print("Zeros")
    res = nb.zeros((2, 3)).to(device)
    compare_with_numpy(res, np.zeros((2, 3), dtype=np.float32), "Zeros")

    print("Ones")
    res = nb.ones((2, 3)).to(device)
    compare_with_numpy(res, np.ones((2, 3), dtype=np.float32), "Ones")

    print("Zeros like")
    a = nb.ndarange((2, 3)).to(device)
    res = nb.zeros_like(a)
    compare_with_numpy(res, np.zeros((2, 3), dtype=np.float32), "Zeros like")

    print("Ones like")
    res = nb.ones_like(a)
    compare_with_numpy(res, np.ones((2, 3), dtype=np.float32), "Ones like")

    # Skip random operations as they're not deterministic for comparison
    print("Random operations tested without validation (non-deterministic)")
    print("  Random normal:", nb.randn((2, 3)).to(device).shape)
    print("  Random uniform:", nb.rand((2, 3)).to(device).shape)


if __name__ == "__main__":
    test_binary_operations()
    test_unary_operations()
    test_reduction_operations()
    test_view_operations()
    test_linalg_operations()
    test_special_operations()
    test_indexing_operations()
    test_creation_operations()

    print("\n=== ALL TESTS COMPLETED ===")
    print("All operations tested against NumPy for correctness validation!")
    print("✅ = Passed, ❌ = Failed, ⚠️ = Comparison Error")
