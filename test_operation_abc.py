"""Test the refactored Operation ABC with flexible signatures.

This file rigorously tests the changes made in the previous conversation:
1. Mixed Tensor/non-Tensor args work correctly
2. Pytree-based input/output mapping (Tensor <-> TensorValue via tree_map)
3. op_args/op_kwargs are stored correctly for traced tensors
4. Chained operations produce correct numerical results via lazy compilation

Uses np.from_dlpack(tensor) for output verification - this is the public
DLPack interface that triggers lazy graph compilation and execution.
"""

import numpy as np
from eager.tensor import Tensor
from eager.ops import Operation
from eager import binary_ops
from max.graph import TensorValue, ops


# =============================================================================
# Tests for Basic Operations and Chaining (Lazy Compilation + Execution)
# =============================================================================

def test_single_op():
    """Test a single operation compiles and executes correctly."""
    print("\n" + "=" * 50)
    print("Test: Single Operation (Compile + Execute)")
    print("=" * 50)
    
    x = Tensor.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    y = Tensor.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    z = binary_ops.add(x, y)
    
    # np.from_dlpack triggers lazy compilation and execution
    result = np.from_dlpack(z)
    expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    
    assert np.allclose(result, expected), f"Mismatch: {result} vs {expected}"
    print(f"  x + y = {result}")
    print("✓ Single op compiles and executes correctly!")


def test_chained_ops():
    """Test multiple operations build correct graph and execute."""
    print("\n" + "=" * 50)
    print("Test: Chained Operations (Graph Building + Execution)")
    print("=" * 50)
    
    # Compute: ((x + 1) * 2) - 3
    x = Tensor.arange(1, 5)  # [1, 2, 3, 4]
    one = Tensor.constant(1.0)
    two = Tensor.constant(2.0)
    three = Tensor.constant(3.0)
    
    step1 = binary_ops.add(x, one)       # [2, 3, 4, 5]
    step2 = binary_ops.mul(step1, two)   # [4, 6, 8, 10]
    step3 = binary_ops.sub(step2, three) # [1, 3, 5, 7]
    
    result = np.from_dlpack(step3)
    expected = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    
    print(f"  ((x + 1) * 2) - 3 = {result}")
    print(f"  expected = {expected}")
    assert np.allclose(result, expected), f"Mismatch: {result} vs {expected}"
    print("✓ Chained ops build correct graph and execute!")


def test_chained_ops_using_operators():
    """Test chaining using Python operators triggers compilation."""
    print("\n" + "=" * 50)
    print("Test: Chained Operations (Python Operators)")
    print("=" * 50)
    
    x = Tensor.arange(0, 5)  # [0, 1, 2, 3, 4]
    
    # Uses __add__, __mul__, __sub__ which call binary_ops internally
    y = (x + 10) * 2 - 5  # [15, 17, 19, 21, 23]
    
    result = np.from_dlpack(y)
    expected = np.array([15.0, 17.0, 19.0, 21.0, 23.0], dtype=np.float32)
    
    print(f"  (x + 10) * 2 - 5 = {result}")
    print(f"  expected = {expected}")
    assert np.allclose(result, expected), f"Mismatch: {result} vs {expected}"
    print("✓ Operator chaining compiles and executes correctly!")


def test_matmul():
    """Test matrix multiplication compilation and execution."""
    print("\n" + "=" * 50)
    print("Test: Matrix Multiplication")
    print("=" * 50)
    
    # 2x3 @ 3x2 = 2x2
    a = Tensor.constant(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    b = Tensor.constant(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    
    c = binary_ops.matmul(a, b)
    
    result = np.from_dlpack(c)
    expected = np.array([[22, 28], [49, 64]], dtype=np.float32)
    
    print(f"  a @ b = \n{result}")
    print(f"  expected = \n{expected}")
    assert np.allclose(result, expected), f"Mismatch: {result} vs {expected}"
    print("✓ Matmul compiles and executes correctly!")


def test_longer_chain():
    """Test a longer computation chain to verify graph building."""
    print("\n" + "=" * 50)
    print("Test: Longer Computation Chain")
    print("=" * 50)
    
    x = Tensor.constant(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    
    # Build a 5-step computation
    r1 = x + 1       # [2, 3, 4, 5]
    r2 = r1 * 2      # [4, 6, 8, 10]
    r3 = r2 - 1      # [3, 5, 7, 9]
    r4 = r3 / 2      # [1.5, 2.5, 3.5, 4.5]
    r5 = r4 + r4     # [3, 5, 7, 9]
    
    result = np.from_dlpack(r5)
    expected = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)
    
    print(f"  ((((x + 1) * 2) - 1) / 2) + self = {result}")
    assert np.allclose(result, expected), f"Mismatch: {result} vs {expected}"
    print("✓ Longer chain compiles and executes correctly!")


# =============================================================================
# Tests for Traced vs Untraced Behavior
# =============================================================================

def test_traced_propagation():
    """Test that traced status propagates through ops."""
    print("\n" + "=" * 50)
    print("Test: Traced Propagation")
    print("=" * 50)
    
    # Untraced: no parent references stored
    x_untraced = Tensor.ones((2, 2))
    y_untraced = binary_ops.add(x_untraced, x_untraced)
    assert not y_untraced._impl.traced, "Untraced input -> untraced output"
    assert len(y_untraced._impl.parents) == 0, "Untraced should not store parents"
    print(f"  Untraced: traced={y_untraced._impl.traced}, parents={len(y_untraced._impl.parents)}")
    
    # Traced: parent references stored
    x_traced = Tensor.ones((2, 2), traced=True)
    y_traced = binary_ops.add(x_traced, x_traced)
    assert y_traced._impl.traced, "Traced input -> traced output"
    assert len(y_traced._impl.parents) > 0, "Traced should store parents"
    print(f"  Traced: traced={y_traced._impl.traced}, parents={len(y_traced._impl.parents)}")
    
    # Mixed: one traced, one not -> output traced
    x_mixed_untraced = Tensor.ones((2, 2))
    y_mixed = binary_ops.add(x_traced, x_mixed_untraced)
    assert y_mixed._impl.traced, "Mixed (one traced) -> traced output"
    print(f"  Mixed: traced={y_mixed._impl.traced}")
    
    print("✓ Traced propagation works!")


def test_op_args_storage():
    """Test that op_args is stored correctly for traced tensors."""
    print("\n" + "=" * 50)
    print("Test: op_args Storage")
    print("=" * 50)
    
    # Untraced: op_args should be None (memory optimization)
    x_untraced = Tensor.ones((2, 2))
    y_untraced = binary_ops.add(x_untraced, x_untraced)
    assert y_untraced._impl.op_args is None, "Untraced should not store op_args"
    print(f"  Untraced op_args: {y_untraced._impl.op_args}")
    
    # Traced: op_args should contain original inputs
    x_traced = Tensor.ones((2, 2), traced=True)
    y_traced = binary_ops.add(x_traced, x_traced)
    assert y_traced._impl.op_args is not None, "Traced should store op_args"
    assert len(y_traced._impl.op_args) == 2, f"Should have 2 args, got {len(y_traced._impl.op_args)}"
    assert y_traced._impl.op_args[0] is x_traced, "First arg should be x_traced"
    assert y_traced._impl.op_args[1] is x_traced, "Second arg should be x_traced" 
    print(f"  Traced op_args: {len(y_traced._impl.op_args)} args")
    print(f"  Args match original inputs: {y_traced._impl.op_args[0] is x_traced}")
    
    print("✓ op_args storage works!")


# =============================================================================
# Tests for Custom Ops with Mixed Args (Non-Tensor positional args)
# =============================================================================

class ReshapeOp(Operation):
    """Custom op with mixed args: Tensor + non-Tensor (shape tuple)."""
    
    @property
    def name(self) -> str:
        return "reshape"
    
    def maxpr(self, x: TensorValue, shape: tuple) -> TensorValue:
        # x is TensorValue (converted from Tensor)
        # shape passes through unchanged (not a Tensor)
        return ops.reshape(x, shape)


def test_mixed_positional_args():
    """Test ops with both Tensor and non-Tensor positional args."""
    print("\n" + "=" * 50)
    print("Test: Mixed Positional Args (Reshape)")
    print("=" * 50)
    
    reshape_op = ReshapeOp()
    
    x = Tensor.arange(0, 12)  # [0, 1, 2, ..., 11]
    shape = (3, 4)  # Non-Tensor positional arg
    
    y = reshape_op(x, shape)
    
    result = np.from_dlpack(y)
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    
    print(f"  x.shape: {tuple(x.shape)}")
    print(f"  reshape(x, {shape}).shape: {tuple(y.shape)}")
    
    assert result.shape == (3, 4), f"Shape mismatch: {result.shape}"
    assert np.allclose(result, expected), f"Value mismatch"
    print("✓ Mixed positional args compile and execute correctly!")


def test_mixed_args_traced_stores_all():
    """Verify traced tensors store non-Tensor args in op_args too."""
    print("\n" + "=" * 50)
    print("Test: Mixed Args in Traced Mode (op_args storage)")
    print("=" * 50)
    
    reshape_op = ReshapeOp()
    
    x = Tensor.arange(0, 6)
    x._impl.traced = True
    shape = (2, 3)
    
    y = reshape_op(x, shape)
    
    # Verify op_args contains both the tensor AND the shape
    assert y._impl.op_args is not None, "Traced should store op_args"
    assert len(y._impl.op_args) == 2, f"Should have 2 args: {len(y._impl.op_args)}"
    assert y._impl.op_args[0] is x, "First arg should be the tensor"
    assert y._impl.op_args[1] == (2, 3), f"Second arg should be shape tuple: {y._impl.op_args[1]}"
    
    print(f"  op_args[0] is Tensor: {isinstance(y._impl.op_args[0], Tensor)}")
    print(f"  op_args[1] is shape tuple: {y._impl.op_args[1]}")
    print("✓ Mixed args stored correctly for VJP/JVP access!")


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" OPERATION ABC TESTS: Lazy Compilation & Execution")
    print("=" * 60)
    
    # Core correctness: lazy compilation + execution
    test_single_op()
    test_chained_ops()
    test_chained_ops_using_operators()
    test_matmul()
    test_longer_chain()
    
    # Tracing behavior (for autograd)
    test_traced_propagation()
    test_op_args_storage()
    
    # Mixed args (new flexible signature)
    test_mixed_positional_args()
    test_mixed_args_traced_stores_all()
    
    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED!")
    print("=" * 60 + "\n")
