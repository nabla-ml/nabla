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
    
    # Untraced: op_args should be empty tuple (memory optimization)
    # Note: OutputRefs is created, but op_args is empty tuple
    x_untraced = Tensor.ones((2, 2))
    y_untraced = binary_ops.add(x_untraced, x_untraced)
    
    # Check via output_refs
    assert y_untraced._impl.output_refs is not None, "Untraced should have OutputRefs"
    refs_untraced = y_untraced._impl.output_refs
    assert refs_untraced.op_args == (), f"Untraced op_args should be empty, got {refs_untraced.op_args}"
    print(f"  Untraced op_args: {refs_untraced.op_args}")
    
    # Traced: op_args should contain original inputs
    x_traced = Tensor.ones((2, 2), traced=True)
    y_traced = binary_ops.add(x_traced, x_traced)
    
    assert y_traced._impl.output_refs is not None, "Traced should have OutputRefs"
    refs_traced = y_traced._impl.output_refs
    op_args = refs_traced.op_args
    
    assert op_args is not None, "Traced should store op_args"
    assert len(op_args) == 2, f"Should have 2 args, got {len(op_args)}"
    assert op_args[0] is x_traced, "First arg should be x_traced"
    assert op_args[1] is x_traced, "Second arg should be x_traced" 
    print(f"  Traced op_args: {len(op_args)} args")
    print(f"  Args match original inputs: {op_args[0] is x_traced}")
    
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
    # Use output_refs to access op_args
    assert y._impl.output_refs is not None, "Traced should have OutputRefs"
    op_args = y._impl.output_refs.op_args
    
    assert op_args is not None, "Traced should store op_args"
    assert len(op_args) == 2, f"Should have 2 args: {len(op_args)}"
    assert op_args[0] is x, "First arg should be the tensor"
    assert op_args[1] == (2, 3), f"Second arg should be shape tuple: {op_args[1]}"
    
    print(f"  op_args[0] is Tensor: {isinstance(op_args[0], Tensor)}")
    print(f"  op_args[1] is shape tuple: {op_args[1]}")
    print("✓ Mixed args stored correctly for VJP/JVP access!")


# =============================================================================
# Tests for Creation Ops (Operation ABC pattern)
# =============================================================================

def test_creation_ops_have_op_attribute():
    """Verify creation ops produce tensors with proper op attribute."""
    print("\n" + "=" * 50)
    print("Test: Creation Ops Have Op Attribute")
    print("=" * 50)
    
    from eager import creation
    
    z = creation.zeros((2, 2))
    o = creation.ones((2, 2))
    a = creation.arange(0, 5)
    c = creation.constant(np.array([1, 2, 3], dtype=np.float32))
    u = creation.uniform((3,))
    g = creation.gaussian((3,))
    
    assert z._impl.op is not None, "zeros should have op"
    assert o._impl.op is not None, "ones should have op"
    assert a._impl.op is not None, "arange should have op"
    assert c._impl.op is not None, "constant should have op"
    assert u._impl.op is not None, "uniform should have op"
    assert g._impl.op is not None, "gaussian should have op"
    
    print(f"  zeros op: {z._impl.op.name}")
    print(f"  ones op: {o._impl.op.name}")
    print(f"  arange op: {a._impl.op.name}")
    print(f"  constant op: {c._impl.op.name}")
    print(f"  uniform op: {u._impl.op.name}")
    print(f"  gaussian op: {g._impl.op.name}")
    
    print("✓ Creation ops have proper op attributes!")


def test_creation_ops_correct_values():
    """Verify creation ops produce correct values."""
    print("\n" + "=" * 50)
    print("Test: Creation Ops Correct Values")
    print("=" * 50)
    
    from eager import creation
    
    z = creation.zeros((3,))
    o = creation.ones((3,))
    a = creation.arange(0, 5)
    f = creation.full((2,), 7.0)
    
    assert np.allclose(np.from_dlpack(z), [0, 0, 0])
    assert np.allclose(np.from_dlpack(o), [1, 1, 1])
    assert np.allclose(np.from_dlpack(a), [0, 1, 2, 3, 4])
    assert np.allclose(np.from_dlpack(f), [7, 7])
    
    print("  zeros: [0, 0, 0] ✓")
    print("  ones: [1, 1, 1] ✓")
    print("  arange(0, 5): [0, 1, 2, 3, 4] ✓")
    print("  full((2,), 7.0): [7, 7] ✓")
    
    print("✓ Creation ops produce correct values!")


def test_tensor_factory_uses_creation_ops():
    """Verify Tensor factory methods use creation ops internally."""
    print("\n" + "=" * 50)
    print("Test: Tensor Factory Uses Creation Ops")
    print("=" * 50)
    
    z = Tensor.zeros((2, 2))
    o = Tensor.ones((2, 2))
    a = Tensor.arange(0, 3)
    
    # Should have op attributes from creation module
    assert z._impl.op is not None, "Tensor.zeros should have op"
    assert o._impl.op is not None, "Tensor.ones should have op"
    assert a._impl.op is not None, "Tensor.arange should have op"
    
    assert z._impl.op.name == "zeros"
    assert o._impl.op.name == "ones"  
    assert a._impl.op.name == "arange"
    
    print(f"  Tensor.zeros -> op.name: {z._impl.op.name}")
    print(f"  Tensor.ones -> op.name: {o._impl.op.name}")
    print(f"  Tensor.arange -> op.name: {a._impl.op.name}")
    
    print("✓ Tensor factory methods use creation ops!")


# =============================================================================
# Tests for Conditional Logic (Triggers Eager Evaluation)
# =============================================================================

def test_item_triggers_evaluation():
    """Test that .item() triggers lazy graph compilation and execution."""
    print("\n" + "=" * 50)
    print("Test: .item() Triggers Evaluation")
    print("=" * 50)
    
    # Build a computation graph
    x = Tensor.constant(5.0)
    y = x * 2 + 3  # Should be 13.0
    
    # .item() should trigger evaluation
    result = y.item()
    
    assert result == 13.0, f"Expected 13.0, got {result}"
    print(f"  x * 2 + 3 = {result}")
    print("✓ .item() triggers evaluation correctly!")


def test_bool_conversion_triggers_evaluation():
    """Test that bool(tensor) triggers evaluation for conditionals."""
    print("\n" + "=" * 50)
    print("Test: bool() Triggers Evaluation")
    print("=" * 50)
    
    # Build computation
    x = Tensor.constant(1.0)
    y = x + 0.5  # 1.5, truthy
    
    # bool() should trigger evaluation
    if y:
        result = "truthy"
    else:
        result = "falsy"
    
    assert result == "truthy", f"Expected truthy, got {result}"
    print(f"  bool(1.5) -> {result}")
    
    # Test falsy case
    z = Tensor.constant(0.0)
    if z:
        result2 = "truthy"
    else:
        result2 = "falsy"
    
    assert result2 == "falsy", f"Expected falsy, got {result2}"
    print(f"  bool(0.0) -> {result2}")
    print("✓ bool() triggers evaluation correctly!")


def test_conditional_branching():
    """Test conditional branching based on computed tensor values."""
    print("\n" + "=" * 50)
    print("Test: Conditional Branching")
    print("=" * 50)
    
    def compute_with_branch(x_val: float) -> str:
        x = Tensor.constant(x_val)
        y = x * x - 4  # x² - 4
        
        # This should trigger evaluation
        if y.item() > 0:
            return "positive"
        elif y.item() < 0:
            return "negative"
        else:
            return "zero"
    
    # x=3: 9-4=5 > 0
    assert compute_with_branch(3.0) == "positive"
    print("  3² - 4 = 5 -> positive ✓")
    
    # x=1: 1-4=-3 < 0
    assert compute_with_branch(1.0) == "negative"
    print("  1² - 4 = -3 -> negative ✓")
    
    # x=2: 4-4=0
    assert compute_with_branch(2.0) == "zero"
    print("  2² - 4 = 0 -> zero ✓")
    
    print("✓ Conditional branching works!")


def test_iterative_computation():
    """Test while-loop style computation using tensor values as conditions."""
    print("\n" + "=" * 50)
    print("Test: Iterative Computation")
    print("=" * 50)
    
    # Compute sum 1 + 2 + 3 + ... until sum >= 10
    total = Tensor.constant(0.0)
    i = 1
    
    while total.item() < 10:
        total = total + Tensor.constant(float(i))
        i += 1
    
    result = total.item()
    # 1+2+3+4 = 10
    assert result == 10.0, f"Expected 10.0, got {result}"
    assert i == 5, f"Expected 5 iterations, got {i}"
    
    print(f"  Sum 1+2+3+4 = {result} (after {i-1} iterations)")
    print("✓ Iterative computation works!")


def test_chained_conditionals():
    """Test multiple evaluations in sequence."""
    print("\n" + "=" * 50)
    print("Test: Chained Conditionals")  
    print("=" * 50)
    
    results = []
    
    for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        x = Tensor.constant(val)
        y = x * 2
        # Each .item() triggers a fresh evaluation
        results.append(y.item())
    
    expected = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert results == expected, f"Expected {expected}, got {results}"
    
    print(f"  [1,2,3,4,5] * 2 = {results}")
    print("✓ Chained conditionals work!")


def test_graph_conditional_graph():
    """Test interleaved: graph -> conditional -> graph -> conditional -> result.
    
    This tests that lazy graphs work correctly between control flow points.
    """
    print("\n" + "=" * 50)
    print("Test: Graph -> Conditional -> Graph -> Conditional")  
    print("=" * 50)
    
    # Phase 1: Build lazy graph
    x = Tensor.constant(3.0)
    y = x * x  # 9.0
    
    # Conditional 1: triggers eval of y
    if y.item() > 5:
        scale = 2.0
    else:
        scale = 0.5
    
    # Phase 2: Build another lazy graph using conditional result
    z = y * scale + 1  # 9*2+1 = 19
    
    # Conditional 2: triggers eval of z
    if z.item() > 10:
        message = "large"
    else:
        message = "small"
    
    # Phase 3: Build final lazy graph
    final = z + 100  # 19+100 = 119
    
    result = final.item()
    
    assert result == 119.0, f"Expected 119.0, got {result}"
    assert message == "large", f"Expected 'large', got {message}"
    
    print(f"  x=3, y=x²=9, scale=2 (since 9>5)")
    print(f"  z=y*scale+1=19, message='large' (since 19>10)")
    print(f"  final=z+100=119 ✓")
    print("✓ Interleaved graph/conditional pattern works!")


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
    
    # Creation ops (Operation ABC pattern)
    test_creation_ops_have_op_attribute()
    test_creation_ops_correct_values()
    test_tensor_factory_uses_creation_ops()
    
    # Conditional logic (triggers eager evaluation)
    test_item_triggers_evaluation()
    test_bool_conversion_triggers_evaluation()
    test_conditional_branching()
    test_iterative_computation()
    test_chained_conditionals()
    test_graph_conditional_graph()
    
    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED!")
    print("=" * 60 + "\n")
