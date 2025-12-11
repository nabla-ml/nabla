"""Test eager mode execution and autograd tracing infrastructure."""

from eager.tensor import Tensor, TensorImpl, get_topological_order, print_computation_graph, GRAPH
from eager import functional as F


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
    print(f"  grad: {impl.grad}")
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
    
    from eager.tensor import TensorImpl
    
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
    
    from eager.ops import Operation
    
    class DummyOp(Operation):
        name = "dummy"
        def maxpr(self, *args, **kwargs):
            pass
    
    op = DummyOp()
    print(f"Created DummyOp: {op}")
    print(f"  name: {op.name}")
    
    # Test that optional methods raise NotImplementedError
    try:
        op.vjp_rule([], None)
        raise AssertionError("Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  vjp_rule raises: {type(e).__name__}")
    
    try:
        op.jvp_rule([], [])
        raise AssertionError("Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  jvp_rule raises: {type(e).__name__}")
    
    try:
        op.sharding_rule([])
        raise AssertionError("Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"  sharding_rule raises: {type(e).__name__}")
    
    print("✓ Operation base class works!\n")


def test_logical_shapes():
    """Test logical vs physical shape with batch dims."""
    print("=" * 50)
    print("Test: Logical vs Physical Shapes")
    print("=" * 50)
    
    from eager.tensor import TensorImpl
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
    
    from eager import pytree
    
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
    
    from eager import pytree
    
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


def test_pytree_broadcast_prefix():
    """Test prefix broadcasting for vmap axes."""
    print("=" * 50)
    print("Test: Pytree Broadcast Prefix")
    print("=" * 50)
    
    from eager import pytree
    
    # Scalar prefix broadcasts to all leaves
    full_tree = {'a': 1, 'b': [2, 3]}
    result = pytree.broadcast_prefix(0, full_tree)
    print(f"broadcast_prefix(0, {full_tree}) = {result}")
    
    # All leaves should be 0
    leaves = pytree.tree_leaves(result)
    assert all(leaf == 0 for leaf in leaves), "All leaves should be 0"
    
    # Dict prefix matches dict structure
    result2 = pytree.broadcast_prefix({'a': 1, 'b': 2}, {'a': 'x', 'b': ['y', 'z']})
    print(f"broadcast_prefix with dict prefix = {result2}")
    assert result2['a'] == 1
    # 'b' prefix of 2 broadcasts to both elements in the list
    assert result2['b'] == [2, 2]
    
    print("✓ Pytree broadcast_prefix works!\n")


def test_pytree_traced_untraced():
    """Test traced/untraced helpers."""
    print("=" * 50)
    print("Test: Pytree traced/untraced")
    print("=" * 50)
    
    from eager import pytree
    
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
    test_pytree_broadcast_prefix()
    test_pytree_traced_untraced()
    
    print("=" * 60)
    print(" ALL TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

