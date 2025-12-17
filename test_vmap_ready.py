"""Test new batch_dims propagation and view operations for vmap support."""

from eager.tensor import Tensor
from eager import view_ops


def test_unsqueeze():
    """Test unsqueeze operation."""
    print("=" * 50)
    print("Test: unsqueeze")
    print("=" * 50)
    
    x = Tensor.ones((3, 4))
    print(f"x.shape: {tuple(x.shape)}")
    
    # Unsqueeze at front
    y = view_ops.unsqueeze(x, axis=0)
    print(f"unsqueeze(x, axis=0).shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (1, 3, 4), f"Expected (1, 3, 4), got {tuple(y.shape)}"
    
    # Unsqueeze at end  
    z = view_ops.unsqueeze(x, axis=-1)
    print(f"unsqueeze(x, axis=-1).shape: {tuple(z.shape)}")
    assert tuple(z.shape) == (3, 4, 1), f"Expected (3, 4, 1), got {tuple(z.shape)}"
    
    print("✓ unsqueeze works!\n")


def test_squeeze():
    """Test squeeze operation."""
    print("=" * 50)
    print("Test: squeeze")
    print("=" * 50)
    
    x = Tensor.ones((1, 3, 4))
    print(f"x.shape: {tuple(x.shape)}")
    
    y = view_ops.squeeze(x, axis=0)
    print(f"squeeze(x, axis=0).shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (3, 4), f"Expected (3, 4), got {tuple(y.shape)}"
    
    print("✓ squeeze works!\n")


def test_swap_axes():
    """Test swap_axes operation."""
    print("=" * 50)
    print("Test: swap_axes")
    print("=" * 50)
    
    x = Tensor.ones((2, 3, 4))
    print(f"x.shape: {tuple(x.shape)}")
    
    y = view_ops.swap_axes(x, axis1=0, axis2=2)
    print(f"swap_axes(x, axis1=0, axis2=2).shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (4, 3, 2), f"Expected (4, 3, 2), got {tuple(y.shape)}"
    
    print("✓ swap_axes works!\n")


def test_moveaxis():
    """Test moveaxis operation."""
    print("=" * 50)
    print("Test: moveaxis")
    print("=" * 50)
    
    x = Tensor.ones((2, 3, 4))
    print(f"x.shape: {tuple(x.shape)}")
    
    # Move axis 0 to position 2
    y = view_ops.moveaxis(x, source=0, destination=2)
    print(f"moveaxis(x, source=0, destination=2).shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (3, 4, 2), f"Expected (3, 4, 2), got {tuple(y.shape)}"
    
    print("✓ moveaxis works!\n")


def test_broadcast_to():
    """Test broadcast_to operation."""
    print("=" * 50)
    print("Test: broadcast_to")
    print("=" * 50)
    
    x = Tensor.ones((3, 1))
    print(f"x.shape: {tuple(x.shape)}")
    
    y = view_ops.broadcast_to(x, shape=(2, 3, 4))
    print(f"broadcast_to(x, shape=(2, 3, 4)).shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (2, 3, 4), f"Expected (2, 3, 4), got {tuple(y.shape)}"
    
    print("✓ broadcast_to works!\n")


def test_incr_decr_batch_dims():
    """Test batch_dims counter operations."""
    print("=" * 50)
    print("Test: incr_batch_dims / decr_batch_dims")
    print("=" * 50)
    
    x = Tensor.ones((2, 3, 4))
    print(f"x.batch_dims: {x.batch_dims}")
    assert x.batch_dims == 0
    
    y = view_ops.incr_batch_dims(x)
    print(f"incr_batch_dims(x).batch_dims: {y.batch_dims}")
    assert y.batch_dims == 1, f"Expected 1, got {y.batch_dims}"
    
    z = view_ops.incr_batch_dims(y)
    print(f"incr_batch_dims(y).batch_dims: {z.batch_dims}")
    assert z.batch_dims == 2, f"Expected 2, got {z.batch_dims}"
    
    w = view_ops.decr_batch_dims(z)
    print(f"decr_batch_dims(z).batch_dims: {w.batch_dims}")
    assert w.batch_dims == 1, f"Expected 1, got {w.batch_dims}"
    
    print("✓ incr/decr_batch_dims works!\n")


def test_move_axis_to_batch_dims():
    """Test moving axis to batch dims."""
    print("=" * 50)
    print("Test: move_axis_to_batch_dims")
    print("=" * 50)
    
    # Shape (2, 3, 4), batch_dims=0, logical=(2,3,4)
    x = Tensor.ones((2, 3, 4))
    print(f"x.shape (logical): {tuple(x.shape)}, x.batch_dims: {x.batch_dims}")
    
    # Move axis 1 (value 3) to batch dims - should move to front and incr batch_dims
    y = view_ops.move_axis_to_batch_dims(x, axis=1)
    print(f"move_axis_to_batch_dims(x, axis=1):")
    print(f"  logical_shape: {tuple(y.shape)}, batch_dims: {y.batch_dims}")
    print(f"  physical_shape: {y._impl.physical_shape}")
    
    # Physical shape should be (3, 2, 4) - the 3 moved to front
    assert y._impl.physical_shape == (3, 2, 4), f"Expected physical (3, 2, 4), got {y._impl.physical_shape}"
    # batch_dims should be 1
    assert y.batch_dims == 1, f"Expected batch_dims=1, got {y.batch_dims}"
    # Logical shape should be (2, 4) - physical[batch_dims:]
    assert tuple(y.shape) == (2, 4), f"Expected logical (2, 4), got {tuple(y.shape)}"
    
    print("✓ move_axis_to_batch_dims works!\n")


def test_move_axis_from_batch_dims():
    """Test moving axis from batch dims to logical shape."""
    print("=" * 50)
    print("Test: move_axis_from_batch_dims")
    print("=" * 50)
    
    # Start with a tensor that has batch_dims=2
    # Physical: (C, B, H, W) = (2, 5, 3, 4)
    # Logical: (H, W) = (3, 4)
    x = Tensor.ones((2, 5, 3, 4))
    x._impl.batch_dims = 2
    print(f"x physical_shape: {x._impl.physical_shape}, x.batch_dims: {x.batch_dims}")
    print(f"  (logical shape: {tuple(x.shape)})")
    
    # Move batch axis 0 (the C=2 dim) to logical position 2 (end of logical)
    y = view_ops.move_axis_from_batch_dims(x, batch_axis=0, logical_destination=2)
    print(f"move_axis_from_batch_dims(x, batch_axis=0, logical_destination=2):")
    print(f"  physical_shape: {y._impl.physical_shape}, batch_dims: {y.batch_dims}")
    print(f"  logical_shape: {tuple(y.shape)}")
    
    # Expected:
    # Physical: (B, H, W, C) = (5, 3, 4, 2)  -- C moved to end
    # batch_dims: 1
    # Logical: (H, W, C) = (3, 4, 2)
    expected_physical = (5, 3, 4, 2)
    expected_batch_dims = 1
    expected_logical = (3, 4, 2)
    
    assert y._impl.physical_shape == expected_physical, f"Expected physical {expected_physical}, got {y._impl.physical_shape}"
    assert y.batch_dims == expected_batch_dims, f"Expected batch_dims={expected_batch_dims}, got {y.batch_dims}"
    assert tuple(y.shape) == expected_logical, f"Expected logical {expected_logical}, got {tuple(y.shape)}"
    
    print("✓ move_axis_from_batch_dims works!\n")


def test_binary_op_batch_dims_propagation():
    """Test that binary ops correctly propagate batch_dims."""
    print("=" * 50)
    print("Test: Binary op batch_dims propagation")
    print("=" * 50)
    
    from eager import binary_ops
    
    # x has batch_dims=1, y has batch_dims=0
    x = Tensor.ones((2, 3, 4))  # physical shape, logically (3, 4) with 1 batch dim
    x._impl.batch_dims = 1
    
    y = Tensor.ones((3, 4))  # physical shape, no batch dims
    y._impl.batch_dims = 0
    
    print(f"x physical_shape: {x._impl.physical_shape}, x.batch_dims: {x.batch_dims}")
    print(f"  x logical_shape: {tuple(x.shape)}")
    print(f"y physical_shape: {y._impl.physical_shape}, y.batch_dims: {y.batch_dims}")
    
    # Add: result should have batch_dims = max(1, 0) = 1
    z = binary_ops.add(x, y)
    print(f"add(x, y):")
    print(f"  physical_shape: {z._impl.physical_shape}, batch_dims: {z.batch_dims}")
    
    assert z.batch_dims == 1, f"Expected batch_dims=1, got {z.batch_dims}"
    
    print("✓ Binary op batch_dims propagation works!\n")


def test_binary_op_traced_broadcasting():
    """Test that traced binary ops correctly broadcast."""
    print("=" * 50)
    print("Test: Traced binary op broadcasting")
    print("=" * 50)
    
    from eager import binary_ops
    
    # Different logical shapes that need broadcasting
    x = Tensor.ones((2, 3, 1), traced=True)
    y = Tensor.ones((4,), traced=True)
    
    print(f"x.shape: {tuple(x.shape)}, x.traced: {x._impl.traced}")
    print(f"y.shape: {tuple(y.shape)}, y.traced: {y._impl.traced}")
    
    # This should broadcast: (2, 3, 1) + (4,) -> (2, 3, 4)
    z = binary_ops.add(x, y)
    print(f"add(x, y):")
    print(f"  shape: {tuple(z.shape)}, traced: {z._impl.traced}")
    
    assert tuple(z.shape) == (2, 3, 4), f"Expected (2, 3, 4), got {tuple(z.shape)}"
    assert z._impl.traced, "Result should be traced"
    
    print("✓ Traced binary op broadcasting works!\n")


def test_vmap_like_workflow():
    """Test a workflow similar to what vmap would do."""
    print("=" * 50)
    print("Test: vmap-like workflow")
    print("=" * 50)
    
    from eager import binary_ops
    
    # Simulate vmap transforming inputs:
    # 1. Take axis and move to front
    # 2. Increment batch_dims
    
    # Original tensor: (5, 3, 4) - user wants to vmap over axis 0
    x = Tensor.ones((5, 3, 4))
    print(f"Original x physical_shape: {x._impl.physical_shape}, batch_dims: {x.batch_dims}")
    
    # Step 1: Move axis 0 to front (already there, so just incr batch_dims)
    x_batched = view_ops.incr_batch_dims(x)
    print(f"After incr_batch_dims:")
    print(f"  physical: {x_batched._impl.physical_shape}, batch_dims={x_batched.batch_dims}")
    print(f"  logical: {tuple(x_batched.shape)}")
    assert x_batched.batch_dims == 1
    
    # Now logical shape is (3, 4) with batch shape (5,)
    logical_shape = tuple(x_batched.shape)
    batch_shape = x_batched._impl.physical_shape[:x_batched.batch_dims]
    print(f"  batch_shape: {batch_shape}, logical_shape: {logical_shape}")
    assert logical_shape == (3, 4)
    assert batch_shape == (5,)
    
    # Do a binary op that should preserve batch_dims
    y = Tensor.ones((3, 1))  # unbatched tensor that broadcasts with logical shape
    z = binary_ops.mul(x_batched, y)
    print(f"After mul with unbatched y:")
    print(f"  physical: {z._impl.physical_shape}, batch_dims={z.batch_dims}")
    print(f"  logical: {tuple(z.shape)}")
    
    assert z.batch_dims == 1, f"Expected batch_dims=1, got {z.batch_dims}"
    
    # Un-vmap: decrement batch_dims
    z_unbatched = view_ops.decr_batch_dims(z)
    print(f"After decr_batch_dims:")
    print(f"  physical: {z_unbatched._impl.physical_shape}, batch_dims={z_unbatched.batch_dims}")
    assert z_unbatched.batch_dims == 0
    
    print("✓ vmap-like workflow works!\n")


def test_move_axis_to_batch_dims_with_existing_batch():
    """Test move_axis_to_batch_dims when there are already batch dims."""
    print("=" * 50)
    print("Test: move_axis_to_batch_dims with existing batch_dims")
    print("=" * 50)
    
    # Physical: (B, H, W, C) = (2, 3, 4, 5), batch_dims=1
    # Logical: (H, W, C) = (3, 4, 5)
    x = Tensor.ones((2, 3, 4, 5))
    x._impl.batch_dims = 1
    print(f"x physical_shape: {x._impl.physical_shape}, batch_dims: {x.batch_dims}")
    print(f"  logical shape: {tuple(x.shape)}")
    
    # Move logical axis 2 (C=5) to batch dims
    y = view_ops.move_axis_to_batch_dims(x, axis=2)
    print(f"move_axis_to_batch_dims(x, axis=2):  # axis=2 is C in logical")
    print(f"  physical_shape: {y._impl.physical_shape}, batch_dims: {y.batch_dims}")
    print(f"  logical_shape: {tuple(y.shape)}")
    
    # Expected:
    # physical_axis = 1 + 2 = 3 (C is at physical position 3)
    # Move to front: (C, B, H, W) = (5, 2, 3, 4)
    # batch_dims = 2
    # logical = (H, W) = (3, 4)
    expected_physical = (5, 2, 3, 4)
    expected_batch_dims = 2
    expected_logical = (3, 4)
    
    assert y._impl.physical_shape == expected_physical, f"Expected physical {expected_physical}, got {y._impl.physical_shape}"
    assert y.batch_dims == expected_batch_dims, f"Expected batch_dims={expected_batch_dims}, got {y.batch_dims}"
    assert tuple(y.shape) == expected_logical, f"Expected logical {expected_logical}, got {tuple(y.shape)}"
    
    print("✓ move_axis_to_batch_dims with existing batch_dims works!\n")


def test_nested_vmap_simulation():
    """Test nested vmap simulation with batch_dims > 1."""
    print("=" * 50)
    print("Test: Nested vmap simulation (batch_dims=2)")
    print("=" * 50)
    
    from eager import binary_ops
    
    # Simulate doubly-vmapped tensor: physical=(B1, B2, H, W) = (2, 3, 4, 5)
    x = Tensor.ones((2, 3, 4, 5))
    x._impl.batch_dims = 2
    print(f"x physical: {x._impl.physical_shape}, batch_dims: {x.batch_dims}")
    print(f"  logical: {tuple(x.shape)}")  # (4, 5)
    
    # Do an op - should preserve batch_dims
    y = binary_ops.mul(x, x)
    print(f"After mul(x, x): batch_dims={y.batch_dims}")
    assert y.batch_dims == 2, f"Expected batch_dims=2, got {y.batch_dims}"
    
    # Add with unbatched tensor
    z = Tensor.ones((4, 5))  # No batch dims
    result = binary_ops.add(y, z)
    print(f"After add with unbatched z: batch_dims={result.batch_dims}")
    assert result.batch_dims == 2, f"Expected batch_dims=2, got {result.batch_dims}"
    
    print("✓ Nested vmap simulation works!\n")


def test_vmap_round_trip():
    """Test moving axis to batch and back."""
    print("=" * 50)
    print("Test: vmap round-trip (to_batch → compute → from_batch)")
    print("=" * 50)
    
    from eager import binary_ops
    
    # Original tensor: (3, 4)
    x = Tensor.ones((3, 4))
    original_shape = tuple(x.shape)
    print(f"Original: shape={original_shape}, batch_dims={x.batch_dims}")
    
    # Move axis 0 to batch (vmap entry)
    x_batched = view_ops.move_axis_to_batch_dims(x, axis=0)
    print(f"After move_axis_to_batch_dims(axis=0):")
    print(f"  physical: {x_batched._impl.physical_shape}, batch_dims={x_batched.batch_dims}")
    print(f"  logical: {tuple(x_batched.shape)}")  # (4,)
    
    assert x_batched.batch_dims == 1
    assert tuple(x_batched.shape) == (4,), f"Expected (4,), got {tuple(x_batched.shape)}"
    
    # Do some computation
    y = binary_ops.mul(x_batched, x_batched)
    print(f"After mul: batch_dims={y.batch_dims}")
    
    # Move back from batch (vmap exit) - put axis back at position 0 in logical
    y_unbatched = view_ops.move_axis_from_batch_dims(y, batch_axis=0, logical_destination=0)
    print(f"After move_axis_from_batch_dims(batch_axis=0, logical_destination=0):")
    print(f"  shape: {tuple(y_unbatched.shape)}, batch_dims={y_unbatched.batch_dims}")
    
    assert y_unbatched.batch_dims == 0
    assert tuple(y_unbatched.shape) == original_shape, f"Expected {original_shape}, got {tuple(y_unbatched.shape)}"
    
    print("✓ vmap round-trip works!\n")


def test_negative_logical_destination():
    """Test negative logical_destination in move_axis_from_batch_dims."""
    print("=" * 50)
    print("Test: Negative logical_destination")
    print("=" * 50)
    
    # Physical: (B, H, W) = (2, 3, 4), batch_dims=1
    x = Tensor.ones((2, 3, 4))
    x._impl.batch_dims = 1
    print(f"x physical: {x._impl.physical_shape}, batch_dims: {x.batch_dims}")
    print(f"  logical: {tuple(x.shape)}")  # (3, 4)
    
    # Move batch axis to end of logical using -1
    y = view_ops.move_axis_from_batch_dims(x, batch_axis=0, logical_destination=-1)
    print(f"move_axis_from_batch_dims(x, batch_axis=0, logical_destination=-1):")
    print(f"  physical: {y._impl.physical_shape}, batch_dims: {y.batch_dims}")
    print(f"  logical: {tuple(y.shape)}")  # should be (3, 4, 2)
    
    # Expected: physical=(3, 4, 2), batch_dims=0, logical=(3, 4, 2)
    expected_physical = (3, 4, 2)
    expected_logical = (3, 4, 2)
    
    assert y._impl.physical_shape == expected_physical, f"Expected physical {expected_physical}, got {y._impl.physical_shape}"
    assert tuple(y.shape) == expected_logical, f"Expected logical {expected_logical}, got {tuple(y.shape)}"
    
    print("✓ Negative logical_destination works!\n")


def test_decr_batch_dims_error():
    """Test that decr_batch_dims fails when batch_dims=0."""
    print("=" * 50)
    print("Test: decr_batch_dims error on batch_dims=0")
    print("=" * 50)
    
    x = Tensor.ones((3, 4))
    assert x.batch_dims == 0
    
    try:
        view_ops.decr_batch_dims(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    
    print("✓ decr_batch_dims error handling works!\n")


def main():
    print("\n" + "=" * 60)
    print(" NEW BATCH_DIMS AND VIEW OPS TESTS")
    print("=" * 60 + "\n")
    
    # View ops tests
    test_unsqueeze()
    test_squeeze()
    test_swap_axes()
    test_moveaxis()
    test_broadcast_to()
    
    # Batch dims management tests
    test_incr_decr_batch_dims()
    test_move_axis_to_batch_dims()
    test_move_axis_from_batch_dims()
    
    # Binary op batch_dims tests
    test_binary_op_batch_dims_propagation()
    test_binary_op_traced_broadcasting()
    
    # Workflow test
    test_vmap_like_workflow()
    
    # NEW: Additional edge case tests
    test_move_axis_to_batch_dims_with_existing_batch()
    test_nested_vmap_simulation()
    test_vmap_round_trip()
    test_negative_logical_destination()
    test_decr_batch_dims_error()
    
    print("=" * 60)
    print(" ALL NEW TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
