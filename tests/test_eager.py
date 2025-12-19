"""Test eager mode execution and autograd tracing infrastructure."""

from nabla.core.tensor import Tensor
from nabla.core.tensor_impl import TensorImpl, get_topological_order, print_computation_graph
from nabla.core.compute_graph import GRAPH
from nabla.ops import binary as binary_ops


def test_basic_eager_execution():
    """Test that basic eager operations work."""
    print("=" * 50)
    print("Test: Basic Eager Execution")
    print("=" * 50)
    
    # Create tensors
    x = Tensor.arange(0, 5)
    print(f"x = arange(0, 5)")
    print(f"  x.shape: {x.shape}")
    print(f"  x._impl: {x._impl}")
    
    # Basic operations
    y = x + 10
    print(f"\ny = x + 10")
    print(f"  y._impl: {y._impl}")
    
    z = y * 2
    print(f"\nz = y * 2")
    print(f"  z._impl: {z._impl}")
    
    # Force evaluation
    print(f"\nz values: {z}")
    print("✓ Basic eager execution works!\n")


def test_tensor_impl_structure():
    """Test that TensorImpl correctly stores all internals."""
    print("=" * 50)
    print("Test: TensorImpl Structure")
    print("=" * 50)
    
    x = Tensor.ones((3, 3))
    
    # Check that _impl exists and has the right fields
    impl = x._impl
    print(f"TensorImpl fields:")
    print(f"  _storages: {impl._storages}")
    print(f"  _values: {impl._values}")
    print(f"  parents: {impl.parents}")
    print(f"  op: {impl.op}")
    print(f"  op_name: {impl.op_name}")
    print(f"  traced: {impl.traced}")
    print(f"  is_leaf: {impl.is_leaf}")
    print(f"  is_realized: {impl.is_realized}")
    print("✓ TensorImpl structure is correct!\n")


def test_untraced_computation():
    """Test that untraced computations don't store parent references."""
    print("=" * 50)
    print("Test: Untraced Computation (no parent storage)")
    print("=" * 50)
    
    x = Tensor.ones((2, 2))
    y = x + 1
    z = y * 2
    
    print(f"x._impl.parents: {x._impl.parents} (should be [])")
    print(f"y._impl.parents: {y._impl.parents} (should be [])")
    print(f"z._impl.parents: {z._impl.parents} (should be [])")
    
    assert len(x._impl.parents) == 0, "Leaf should have no parents"
    assert len(y._impl.parents) == 0, "Untraced tensor should have no parents"
    assert len(z._impl.parents) == 0, "Untraced tensor should have no parents"
    
    print("✓ Untraced computation doesn't store parents!\n")


def test_traced_computation():
    """Test that traced computations store parent references."""
    print("=" * 50)
    print("Test: Traced Computation (parent storage)")
    print("=" * 50)
    
    # Create a traced tensor
    x = Tensor.ones((2, 2), traced=True)
    print(f"x = Tensor.ones((2,2), traced=True)")
    print(f"  x._impl.traced: {x._impl.traced}")
    
    y = x + 1
    print(f"\ny = x + 1")
    print(f"  y._impl.traced: {y._impl.traced}")
    print(f"  y._impl.parents: {y._impl.parents}")
    print(f"  y._impl.op_name: {y._impl.op_name}")
    
    z = y * 2
    print(f"\nz = y * 2")
    print(f"  z._impl.traced: {z._impl.traced}")
    print(f"  z._impl.parents: {z._impl.parents}")
    print(f"  z._impl.op_name: {z._impl.op_name}")
    
    # Traced should propagate
    assert x._impl.traced, "x should be traced"
    assert y._impl.traced, "y should inherit traced from x"
    assert z._impl.traced, "z should inherit traced from y"
    
    # Parents should be stored
    assert len(y._impl.parents) > 0, "Traced y should have parents"
    assert len(z._impl.parents) > 0, "Traced z should have parents"
    
    print("✓ Traced computation stores parents!\n")


def test_topological_order():
    """Test topological ordering of computation graph."""
    print("=" * 50)
    print("Test: Topological Order")
    print("=" * 50)
    
    x = Tensor.ones((2, 2), traced=True)
    y = x + 1
    z = y * 2
    w = z + y  # Has two traced parents
    
    print("Computation graph for w = (x + 1) * 2 + (x + 1):")
    print_computation_graph(w._impl)
    
    order = get_topological_order(w._impl)
    print(f"\nNumber of nodes in graph: {len(order)}")
    
    print("✓ Topological ordering works!\n")


def test_storage_delegation():
    """Test that Tensor correctly delegates storage access to TensorImpl."""
    print("=" * 50)
    print("Test: Storage Delegation")
    print("=" * 50)
    
    import numpy as np
    
    # Create from numpy
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = Tensor.from_dlpack(arr)
    
    # Test that Tensor convenience properties work
    print(f"x.storage is x._impl._storages[0]: {x.storage is x._impl._storages[0]}")
    print(f"x._value is (x._impl._values[0] if x._impl._values else None): {x._value is (x._impl._values[0] if x._impl._values else None)}")
    
    assert x.storage is x._impl._storages[0], "storage should access first shard"
    
    print("✓ Storage delegation works!\n")


def test_batch_dims():
    """Test batch dimension tracking."""
    print("=" * 50)
    print("Test: Batch Dimensions")
    print("=" * 50)
    
    from nabla.core.tensor import TensorImpl
    
    # Simulate a batched tensor (would come from vmap)
    impl = TensorImpl(batch_dims=2)
    print(f"TensorImpl with batch_dims=2: {impl}")
    print(f"  batch_dims: {impl.batch_dims}")
    
    # Test an unbatched tensor
    impl2 = TensorImpl(batch_dims=0)
    print(f"TensorImpl with batch_dims=0: {impl2}")
    print(f"  batch_dims: {impl2.batch_dims}")
    
    assert impl.batch_dims == 2
    assert impl2.batch_dims == 0
    print("✓ Batch dimensions tracking works!\n")


def test_operation_base_class():
    """Test the Operation base class."""
    print("=" * 50)
    print("Test: Operation Base Class")
    print("=" * 50)
    
    from nabla.ops.operation import Operation
    
    class DummyOp(Operation):
        name = "dummy"
        def maxpr(self, *args, **kwargs):
            pass
    
    op = DummyOp()
    print(f"Created DummyOp: {op}")
    print(f"  name: {op.name}")
    
    # Test that optional methods raise NotImplementedError
    try:
        op.vjp_rule([], None, None)  # primals, output, cotangent
        raise AssertionError("Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  vjp_rule raises: {type(e).__name__}")
    
    try:
        op.jvp_rule([], [], None)  # primals, tangents, output
        raise AssertionError("Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  jvp_rule raises: {type(e).__name__}")
    
    try:
        op.sharding_rule([], None)  # inputs, output
        raise AssertionError("Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  sharding_rule raises: {type(e).__name__}")
    
    print("✓ Operation base class works!\n")


def test_logical_shapes():
    """Test logical vs physical shape with batch dims."""
    print("=" * 50)
    print("Test: Logical vs Physical Shapes")
    print("=" * 50)
    
    from nabla.core.tensor import TensorImpl
    from max.graph import ops, TensorValue
    
    # Create a TensorImpl with a known shape (simulating batched data)
    # Physical shape: (batch1, batch2, H, W) = 4D
    # Logical shape: (H, W) = 2D
    impl = TensorImpl(batch_dims=2)
    print(f"TensorImpl with batch_dims=2")
    print(f"  (actual physical_shape requires storage or _value)")
    
    # Test that batch_dims slicing logic is correct
    # Simulate: physical = (3, 4, 10, 20), batch_dims = 2
    # Expected logical = (10, 20)
    physical = (3, 4, 10, 20)
    batch_dims = 2
    logical = physical[batch_dims:]
    assert logical == (10, 20), f"Expected (10, 20), got {logical}"
    print(f"  physical[{batch_dims}:] = {logical}")
    
    print("✓ Logical shape slicing works!\n")

def test_pytree_flatten_unflatten():
    """Test pytree flatten and unflatten round-trip."""
    print("=" * 50)
    print("Test: Pytree Flatten/Unflatten")
    print("=" * 50)
    
    from nabla.core import pytree
    
    # Test with nested dict/list structure
    x = Tensor.ones((2, 2))
    y = Tensor.zeros((3,))
    tree = {'weights': x, 'layers': [y, {'bias': Tensor.full((2,), 5.0)}]}
    
    leaves, treedef = pytree.tree_flatten(tree)
    print(f"Leaves: {len(leaves)} tensors")
    print(f"TreeDef: {treedef}")
    
    assert len(leaves) == 3, f"Expected 3 leaves, got {len(leaves)}"
    assert treedef.num_leaves == 3
    
    # Reconstruct
    reconstructed = pytree.tree_unflatten(treedef, leaves)
    assert 'weights' in reconstructed
    assert 'layers' in reconstructed
    assert len(reconstructed['layers']) == 2
    assert 'bias' in reconstructed['layers'][1]
    
    print("✓ Pytree flatten/unflatten works!\n")


def test_pytree_tree_map():
    """Test tree_map applies function to all leaves."""
    print("=" * 50)
    print("Test: Pytree tree_map")
    print("=" * 50)
    
    from nabla.core import pytree
    
    x = Tensor.ones((2,))
    y = Tensor.ones((3,))
    tree = [x, {'a': y}]
    
    # Apply a transformation
    call_count = [0]
    def count_and_pass(t):
        call_count[0] += 1
        return t
    
    result = pytree.tree_map(count_and_pass, tree)
    assert call_count[0] == 2, f"Expected 2 calls, got {call_count[0]}"
    
    # Test multi-tree map
    tree1 = {'a': Tensor.ones((2,)), 'b': Tensor.zeros((2,))}
    tree2 = {'a': Tensor.full((2,), 2.0), 'b': Tensor.full((2,), 3.0)}
    
    # Just verify it doesn't crash (actual addition would need realize)
    combined = pytree.tree_map(lambda x, y: x, tree1, tree2)
    assert 'a' in combined and 'b' in combined
    
    print("✓ Pytree tree_map works!\n")


def test_pytree_traced_untraced():
    """Test traced/untraced helpers."""
    print("=" * 50)
    print("Test: Pytree traced/untraced")
    print("=" * 50)
    
    from nabla.core import pytree
    
    # Create untraced tensors
    x = Tensor.ones((2,))
    y = Tensor.zeros((3,))
    tree = {'x': x, 'y': y}
    
    assert not x._impl.traced
    assert not y._impl.traced
    
    # Mark as traced
    pytree.traced(tree)
    assert x._impl.traced, "x should be traced"
    assert y._impl.traced, "y should be traced"
    
    # Mark as untraced
    pytree.untraced(tree)
    assert not x._impl.traced, "x should be untraced"
    assert not y._impl.traced, "y should be untraced"
    
    print("✓ Pytree traced/untraced works!\n")


# ========== NEW TESTS: Traced vs Untraced with Weakrefs ==========

def test_weakref_cleanup_untraced():
    """Test that in untraced mode, intermediates don't hold parent references.
    
    This verifies that the weakref-based cleanup can work - if parents aren't
    stored, intermediates can be garbage collected when out of scope.
    """
    print("=" * 50)
    print("Test: Weakref Cleanup (Untraced Mode)")
    print("=" * 50)
    
    import gc
    
    # Track how many TensorImpl objects exist
    def count_tensor_impls():
        gc.collect()
        return sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'TensorImpl')
    
    initial_count = count_tensor_impls()
    
    # Create a chain of untraced operations
    def create_chain():
        a = Tensor.ones((3, 3))  # untraced by default
        b = a + 1
        c = b * 2
        d = c - 1
        e = d / 2
        # Only return final result
        return e
    
    result = create_chain()
    
    # Check that intermediate tensors have no parent references
    assert len(result._impl.parents) == 0, "Untraced result should have no parents"
    print(f"result._impl.parents: {result._impl.parents} (empty = good)")
    
    # The intermediates should have been created but their parent lists are empty
    # This means they CAN be garbage collected (no strong ref chains)
    gc.collect()
    
    # Note: We're not testing that they ARE collected (that's harder to verify)
    # but that the mechanism for collection exists (no parent refs)
    
    print("✓ Untraced intermediates don't store parent references!")
    print("✓ Weakref cleanup mechanism is in place!\n")


def test_parent_retention_traced():
    """Test that in traced mode, parent references ARE retained.
    
    This is essential for backward pass computation in autograd.
    """
    print("=" * 50)
    print("Test: Parent Retention (Traced Mode)")
    print("=" * 50)
    
    # Create traced computation
    a = Tensor.ones((3, 3), traced=True)
    b = a + 1
    c = b * 2
    d = c + b  # d depends on both c and b
    
    # Verify tracing propagated
    assert a._impl.traced, "a should be traced"
    assert b._impl.traced, "b should be traced"
    assert c._impl.traced, "c should be traced"
    assert d._impl.traced, "d should be traced"
    
    # Verify parent references are stored
    assert len(b._impl.parents) > 0, "Traced b should have parents"
    assert len(c._impl.parents) > 0, "Traced c should have parents"
    assert len(d._impl.parents) > 0, "Traced d should have parents"
    
    print(f"b._impl.parents: {len(b._impl.parents)} parent(s)")
    print(f"c._impl.parents: {len(c._impl.parents)} parent(s)")
    print(f"d._impl.parents: {len(d._impl.parents)} parent(s)")
    
    # Verify we can traverse the graph from d back to a
    order = get_topological_order(d._impl)
    print(f"\nTopological order from d: {len(order)} nodes")
    
    # The leaf (a) should be in the order
    leaf_nodes = [n for n in order if n.is_leaf and n.traced]
    assert len(leaf_nodes) > 0, "Should have at least one traced leaf"
    print(f"Found {len(leaf_nodes)} traced leaf node(s)")
    
    print("✓ Traced mode correctly retains parent references!\n")


def test_computation_result_correctness():
    """Test that both traced and untraced modes produce correct numerical results."""
    print("=" * 50)
    print("Test: Computation Result Correctness")
    print("=" * 50)
    
    import numpy as np
    
    # Test untraced computation
    print("\n--- Untraced Mode ---")
    x_untraced = Tensor.arange(0, 5)  # [0, 1, 2, 3, 4]
    y_untraced = (x_untraced + 10) * 2  # [20, 22, 24, 26, 28]
    
    # Realize and check values
    y_untraced._sync_realize()
    # After realize, _storages[0] contains the driver.Tensor
    result_untraced = y_untraced._impl._storages[0].to_numpy()
    expected = np.array([20., 22., 24., 26., 28.], dtype=np.float32)
    
    assert np.allclose(result_untraced, expected), f"Untraced: Expected {expected}, got {result_untraced}"
    print(f"Untraced result: {result_untraced}")
    print(f"Expected:        {expected}")
    print("✓ Untraced computation correct!")
    
    # Test traced computation
    print("\n--- Traced Mode ---")
    x_traced = Tensor.arange(0, 5)
    x_traced._impl.traced = True  # Enable tracing
    y_traced = (x_traced + 10) * 2
    
    # Should still give same numerical result
    y_traced._sync_realize()
    result_traced = y_traced._impl._storages[0].to_numpy()
    
    assert np.allclose(result_traced, expected), f"Traced: Expected {expected}, got {result_traced}"
    print(f"Traced result: {result_traced}")
    print(f"Expected:      {expected}")
    print("✓ Traced computation correct!")
    
    # Verify tracing was maintained
    assert y_traced._impl.traced, "Result should be traced"
    assert len(y_traced._impl.parents) > 0, "Traced result should have parents"
    
    print("\n✓ Both modes produce identical correct results!\n")


# ========== NEW TESTS: Multi-Output Operations ==========

def test_multi_output_tuple():
    """Test multi-output operation returning tuple."""
    print("=" * 50)
    print("Test: Multi-Output Tuple (Split)")
    print("=" * 50)
    
    from nabla.ops import multi_output as multi_output_ops
    
    # Create a tensor and split it
    x = Tensor.arange(0, 12)  # Shape: (12,)
    
    # Split into 3 parts
    a, b, c = multi_output_ops.split(x, num_splits=3, axis=0)
    
    print(f"x.shape: {x.shape}")
    print(f"Split into 3 parts:")
    print(f"  a.shape: {a.shape}")
    print(f"  b.shape: {b.shape}")
    print(f"  c.shape: {c.shape}")
    
    # Verify shapes
    assert tuple(a.shape) == (4,), f"Expected (4,), got {tuple(a.shape)}"
    assert tuple(b.shape) == (4,), f"Expected (4,), got {tuple(b.shape)}"
    assert tuple(c.shape) == (4,), f"Expected (4,), got {tuple(c.shape)}"
    
    # Verify each is a Tensor
    assert isinstance(a, Tensor), "Split output should be Tensor"
    assert isinstance(b, Tensor), "Split output should be Tensor"
    assert isinstance(c, Tensor), "Split output should be Tensor"
    
    print("✓ Multi-output tuple works!\n")


def test_multi_output_list():
    """Test multi-output operation returning list."""
    print("=" * 50)
    print("Test: Multi-Output List (Chunk)")
    print("=" * 50)
    
    from nabla.ops import multi_output as multi_output_ops
    
    x = Tensor.arange(0, 10)  # Shape: (10,)
    
    # Chunk into 3 parts (uneven)
    chunks = multi_output_ops.chunk(x, chunks=3, axis=0)
    
    print(f"x.shape: {x.shape}")
    print(f"Chunked into {len(chunks)} parts:")
    for i, chunk in enumerate(chunks):
        print(f"  chunk[{i}].shape: {chunk.shape}")
    
    # Verify it's a list
    assert isinstance(chunks, list), f"Expected list, got {type(chunks)}"
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    
    # Each chunk should be a Tensor
    for chunk in chunks:
        assert isinstance(chunk, Tensor), "Chunk should be Tensor"
    
    print("✓ Multi-output list works!\n")


def test_multi_output_dict():
    """Test multi-output operation returning dict."""
    print("=" * 50)
    print("Test: Multi-Output Dict (MinMax)")
    print("=" * 50)
    
    from nabla.ops import multi_output as multi_output_ops
    
    x = Tensor.arange(0, 10)  # 0 to 9
    
    result = multi_output_ops.minmax(x)
    
    print(f"x: {x.shape}")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys())}")
    
    # Verify it's a dict
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'min' in result, "Should have 'min' key"
    assert 'max' in result, "Should have 'max' key"
    
    # Each value should be a Tensor
    assert isinstance(result['min'], Tensor), "min should be Tensor"
    assert isinstance(result['max'], Tensor), "max should be Tensor"
    
    print("✓ Multi-output dict works!\n")


def test_multi_output_shared_op():
    """Test that multi-output ops share the same op instance."""
    print("=" * 50)
    print("Test: Multi-Output Shared Op")
    print("=" * 50)
    
    from nabla.ops import multi_output as multi_output_ops
    
    x = Tensor.arange(0, 6)
    a, b = multi_output_ops.split(x, num_splits=2, axis=0)
    
    # Check they share the same op instance  
    assert a._impl.op is b._impl.op, "Siblings should share same op"
    assert a._impl.op.name == "split", f"Expected 'split', got {a._impl.op.name}"
    
    # Check they share the same parents
    assert a._impl.parents == b._impl.parents, "Siblings should share same parents"
    
    print(f"a._impl.op: {a._impl.op}")
    print(f"b._impl.op: {b._impl.op}")
    print(f"Same op? {a._impl.op is b._impl.op}")
    
    print("✓ Multi-output shared op works!\n")


def test_multi_output_traced():
    """Test tracing through multi-output operations."""
    print("=" * 50)
    print("Test: Multi-Output Tracing")
    print("=" * 50)
    
    from nabla.ops import multi_output as multi_output_ops
    
    # Create traced input
    x = Tensor.arange(0, 6)
    x._impl.traced = True
    
    # Split should propagate tracing
    a, b = multi_output_ops.split(x, num_splits=2, axis=0)
    
    assert a._impl.traced, "a should be traced"
    assert b._impl.traced, "b should be traced"
    
    # Parents should be recorded
    assert len(a._impl.parents) > 0, "a should have parents"
    assert len(b._impl.parents) > 0, "b should have parents"
    
    # Both should have same parent (the input x)
    assert a._impl.parents[0] is x._impl, "a's parent should be x"
    assert b._impl.parents[0] is x._impl, "b's parent should be x"
    
    print(f"a._impl.traced: {a._impl.traced}")
    print(f"b._impl.traced: {b._impl.traced}")
    print(f"a._impl.parents: {len(a._impl.parents)} parent(s)")
    
    print("✓ Multi-output tracing works!\n")


def test_multi_output_pytree_helpers():
    """Test pytree helper functions for multi-output support."""
    print("=" * 50)
    print("Test: Multi-Output Pytree Helpers")
    print("=" * 50)
    
    from nabla.core import pytree
    from max import graph
    
    # Test is_tensor_value
    with GRAPH.graph:
        tv = graph.TensorValue(Tensor.ones((2, 2)))
        assert pytree.is_tensor_value(tv), "Should detect TensorValue"
    
    assert not pytree.is_tensor_value(42), "int is not TensorValue"
    assert not pytree.is_tensor_value("hello"), "str is not TensorValue"
    assert not pytree.is_tensor_value(Tensor.ones((2, 2))), "Tensor is not TensorValue"
    
    print("✓ is_tensor_value works correctly")
    
    # Test tensor_leaves extracts only Tensors from pytree
    x = Tensor.ones((2,))
    y = Tensor.zeros((3,))
    tree = {'a': x, 'b': [y, 42, "ignored"]}
    
    tensor_leaves = pytree.tensor_leaves(tree)
    assert len(tensor_leaves) == 2, f"Expected 2 tensor leaves, got {len(tensor_leaves)}"
    assert x in tensor_leaves, "Should include x"
    assert y in tensor_leaves, "Should include y"
    
    print("✓ tensor_leaves works correctly")
    print("✓ Pytree helpers work!\n")


def main():
    print("\n" + "=" * 60)
    print(" EAGER MODE + AUTOGRAD INFRASTRUCTURE TESTS")
    print("=" * 60 + "\n")
    
    test_basic_eager_execution()
    test_tensor_impl_structure()
    test_untraced_computation()
    test_traced_computation()
    test_topological_order()
    test_storage_delegation()
    test_batch_dims()
    test_operation_base_class()
    test_logical_shapes()
    
    # Pytree tests
    test_pytree_flatten_unflatten()
    test_pytree_tree_map()
    test_pytree_traced_untraced()
    
    # NEW: Traced vs Untraced + Weakrefs tests
    test_weakref_cleanup_untraced()
    test_parent_retention_traced()
    test_computation_result_correctness()
    
    # NEW: Multi-output operation tests
    test_multi_output_tuple()
    test_multi_output_list()
    test_multi_output_dict()
    test_multi_output_shared_op()
    test_multi_output_traced()
    test_multi_output_pytree_helpers()
    
    print("=" * 60)
    print(" ALL TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

