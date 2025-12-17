"""Test vmap transform for the eager module."""

from eager.tensor import Tensor
from eager.vmap_trafo import vmap
from eager import binary_ops


def test_vmap_basic_square():
    """Test basic vmap over a simple unary function."""
    print("=" * 50)
    print("Test: vmap basic square")
    print("=" * 50)
    
    def square(x):
        return x * x
    
    # Batch of 5 elements
    x = Tensor.arange(1, 6)  # [1, 2, 3, 4, 5]
    print(f"x.shape: {tuple(x.shape)}")
    
    # vmap should map square over axis 0
    vmapped_square = vmap(square)
    result = vmapped_square(x)
    
    print(f"vmap(square)(x).shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (5,), f"Expected (5,), got {tuple(result.shape)}"
    
    print("✓ vmap basic square works!\n")


def test_vmap_binary_same_axes():
    """Test vmap with two inputs batched on the same axis."""
    print("=" * 50)
    print("Test: vmap binary same axes")
    print("=" * 50)
    
    def add(x, y):
        return x + y
    
    x = Tensor.ones((3, 4))  # batch of 3
    y = Tensor.ones((3, 4))  # batch of 3
    print(f"x.shape: {tuple(x.shape)}, y.shape: {tuple(y.shape)}")
    
    # Both batched on axis 0
    result = vmap(add, in_axes=(0, 0))(x, y)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (3, 4), f"Expected (3, 4), got {tuple(result.shape)}"
    
    print("✓ vmap binary same axes works!\n")


def test_vmap_broadcast_none_axis():
    """Test vmap with one input broadcasted (in_axes=None)."""
    print("=" * 50)
    print("Test: vmap broadcast (in_axes=None)")
    print("=" * 50)
    
    def add(x, y):
        return x + y
    
    x = Tensor.ones((5, 3))  # batch of 5
    y = Tensor.full((3,), 10.0)  # scalar to broadcast
    print(f"x.shape: {tuple(x.shape)}, y.shape: {tuple(y.shape)}")
    
    # x batched on axis 0, y broadcasted
    result = vmap(add, in_axes=(0, None))(x, y)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (5, 3), f"Expected (5, 3), got {tuple(result.shape)}"
    
    print("✓ vmap broadcast works!\n")


def test_vmap_out_axes_nonzero():
    """Test vmap with out_axes != 0."""
    print("=" * 50)
    print("Test: vmap out_axes=1")
    print("=" * 50)
    
    def double(x):
        return x * 2
    
    x = Tensor.ones((3, 4))  # batch of 3, logical (4,)
    print(f"x.shape: {tuple(x.shape)}")
    
    # Batch axis should end up at position 1 in output
    result = vmap(double, in_axes=0, out_axes=1)(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (4, 3), f"Expected (4, 3), got {tuple(result.shape)}"
    
    print("✓ vmap out_axes=1 works!\n")


def test_vmap_multi_output():
    """Test vmap with function returning multiple outputs."""
    print("=" * 50)
    print("Test: vmap multi-output")
    print("=" * 50)
    
    def split_process(x):
        return x * 2, x + 1
    
    x = Tensor.ones((5, 3))  # batch of 5
    print(f"x.shape: {tuple(x.shape)}")
    
    result1, result2 = vmap(split_process)(x)
    
    print(f"result1.shape: {tuple(result1.shape)}")
    print(f"result2.shape: {tuple(result2.shape)}")
    assert tuple(result1.shape) == (5, 3), f"Expected (5, 3), got {tuple(result1.shape)}"
    assert tuple(result2.shape) == (5, 3), f"Expected (5, 3), got {tuple(result2.shape)}"
    
    print("✓ vmap multi-output works!\n")


def test_vmap_decorator():
    """Test vmap as a decorator."""
    print("=" * 50)
    print("Test: vmap as decorator")
    print("=" * 50)
    
    @vmap
    def process(x):
        return x * x + 1
    
    x = Tensor.arange(1, 5)  # [1, 2, 3, 4]
    print(f"x.shape: {tuple(x.shape)}")
    
    result = process(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (4,), f"Expected (4,), got {tuple(result.shape)}"
    
    print("✓ vmap decorator works!\n")


def test_vmap_decorator_with_args():
    """Test vmap as a decorator with arguments."""
    print("=" * 50)
    print("Test: vmap decorator with args")
    print("=" * 50)
    
    @vmap(in_axes=1, out_axes=1)
    def process(x):
        return x + 1
    
    x = Tensor.ones((3, 5))  # Shape (3, 5), batch along axis 1
    print(f"x.shape: {tuple(x.shape)}")
    
    result = process(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    # Input (3, 5) with in_axes=1: batch is 5, logical is (3,)
    # Output with out_axes=1: should be (3, 5)
    assert tuple(result.shape) == (3, 5), f"Expected (3, 5), got {tuple(result.shape)}"
    
    print("✓ vmap decorator with args works!\n")


def test_vmap_nested():
    """Test nested vmap (vmap of vmap)."""
    print("=" * 50)
    print("Test: nested vmap")
    print("=" * 50)
    
    def add_one(x):
        return x + 1
    
    x = Tensor.ones((2, 3, 4))  # Two nested batch dims
    print(f"x.shape: {tuple(x.shape)}")
    
    # Double vmap: first over outer, then over middle
    result = vmap(vmap(add_one))(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (2, 3, 4), f"Expected (2, 3, 4), got {tuple(result.shape)}"
    
    print("✓ nested vmap works!\n")


def test_vmap_triple_nested():
    """Test triple nested vmap (vmap of vmap of vmap)."""
    print("=" * 50)
    print("Test: triple nested vmap")
    print("=" * 50)
    
    def scale(x):
        return x * 2
    
    x = Tensor.ones((2, 3, 4, 5))  # Three nested batch dims
    print(f"x.shape: {tuple(x.shape)}")
    
    # Triple vmap
    result = vmap(vmap(vmap(scale)))(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (2, 3, 4, 5), f"Expected (2, 3, 4, 5), got {tuple(result.shape)}"
    
    print("✓ triple nested vmap works!\n")


def test_vmap_pytree_inputs():
    """Test vmap with pytree (dict) inputs."""
    print("=" * 50)
    print("Test: vmap with pytree inputs")
    print("=" * 50)
    
    def process_dict(inputs):
        return inputs['a'] + inputs['b']
    
    inputs = {
        'a': Tensor.ones((5, 3)),  # batch of 5
        'b': Tensor.ones((5, 3)),  # batch of 5
    }
    print(f"inputs['a'].shape: {tuple(inputs['a'].shape)}")
    print(f"inputs['b'].shape: {tuple(inputs['b'].shape)}")
    
    # Pass dict as single arg - in_axes=0 broadcasts to all tensor leaves
    result = vmap(process_dict)(inputs)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (5, 3), f"Expected (5, 3), got {tuple(result.shape)}"
    
    print("✓ vmap pytree inputs works!\n")


def test_vmap_same_shape_broadcast():
    """Test vmap scalar in_axes broadcasts to all args with same shapes."""
    print("=" * 50)
    print("Test: vmap scalar axis broadcast")
    print("=" * 50)
    
    def add_three(x, y, z):
        return x + y + z
    
    # All same batch size and logical shape
    x = Tensor.ones((4, 3))
    y = Tensor.ones((4, 3))
    z = Tensor.ones((4, 3))
    print(f"x.shape: {tuple(x.shape)}, y.shape: {tuple(y.shape)}, z.shape: {tuple(z.shape)}")
    
    # in_axes=0 broadcasts to all three args
    result = vmap(add_three, in_axes=0)(x, y, z)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (4, 3), f"Expected (4, 3), got {tuple(result.shape)}"
    
    print("✓ vmap scalar axis broadcast works!\n")


def test_vmap_negative_axis():
    """Test vmap with negative axis."""
    print("=" * 50)
    print("Test: vmap negative axis")
    print("=" * 50)
    
    def double(x):
        return x * 2
    
    x = Tensor.ones((3, 5))  # batch on last axis
    print(f"x.shape: {tuple(x.shape)}")
    
    # in_axes=-1 means batch is axis 1 (last axis)
    result = vmap(double, in_axes=-1)(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    # Input (3, 5) with in_axes=-1 (axis 1): batch size 5, logical is (3,)
    # Output with out_axes=0: should be (5, 3)
    assert tuple(result.shape) == (5, 3), f"Expected (5, 3), got {tuple(result.shape)}"
    
    print("✓ vmap negative axis works!\n")


def test_vmap_preserves_batch_dims():
    """Test that vmap correctly manages batch_dims counter."""
    print("=" * 50)
    print("Test: vmap preserves batch_dims")
    print("=" * 50)
    
    def identity(x):
        # Inside vmap, tensor should have batch_dims > 0
        print(f"  Inside vmap: x.batch_dims = {x.batch_dims}")
        return x
    
    x = Tensor.ones((4, 3))
    print(f"Before vmap: x.batch_dims = {x.batch_dims}")
    assert x.batch_dims == 0
    
    result = vmap(identity)(x)
    
    print(f"After vmap: result.batch_dims = {result.batch_dims}")
    assert result.batch_dims == 0, "Result should have batch_dims=0 after unbatching"
    
    print("✓ vmap preserves batch_dims works!\n")


def test_vmap_error_inconsistent_batch_sizes():
    """Test that vmap raises error for inconsistent batch sizes."""
    print("=" * 50)
    print("Test: vmap error on inconsistent batch sizes")
    print("=" * 50)
    
    def add(x, y):
        return x + y
    
    x = Tensor.ones((5, 3))  # batch of 5
    y = Tensor.ones((7, 3))  # batch of 7 (inconsistent!)
    
    try:
        vmap(add, in_axes=(0, 0))(x, y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    
    print("✓ vmap error handling works!\n")


def test_vmap_error_axis_out_of_bounds():
    """Test that vmap raises error for axis out of bounds."""
    print("=" * 50)
    print("Test: vmap error on axis out of bounds")
    print("=" * 50)
    
    def double(x):
        return x * 2
    
    x = Tensor.ones((3, 4))  # 2D tensor
    
    try:
        vmap(double, in_axes=5)(x)  # axis 5 doesn't exist
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    
    print("✓ vmap axis bounds checking works!\n")


def test_vmap_axis_size_broadcast():
    """Test vmap with axis_size for pure broadcast."""
    print("=" * 50)
    print("Test: vmap axis_size for pure broadcast")
    print("=" * 50)
    
    def add_one(x):
        return x + 1
    
    scalar = Tensor.full((3,), 5.0)
    print(f"scalar.shape: {tuple(scalar.shape)}")
    
    # All in_axes=None, must provide axis_size
    result = vmap(add_one, in_axes=None, axis_size=4)(scalar)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (4, 3), f"Expected (4, 3), got {tuple(result.shape)}"
    
    print("✓ vmap axis_size works!\n")


def test_vmap_dict_in_axes():
    """Test vmap with dict axis specification for pytree inputs."""
    print("=" * 50)
    print("Test: vmap dict in_axes")
    print("=" * 50)
    
    def process(params):
        return params['w'] + params['b']
    
    params = {
        'w': Tensor.ones((5, 3)),  # batch of 5
        'b': Tensor.full((3,), 1.0),  # broadcast
    }
    print(f"params['w'].shape: {tuple(params['w'].shape)}")
    print(f"params['b'].shape: {tuple(params['b'].shape)}")
    
    # Different axes per key
    result = vmap(process, in_axes={'w': 0, 'b': None})(params)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (5, 3), f"Expected (5, 3), got {tuple(result.shape)}"
    
    print("✓ vmap dict in_axes works!\n")


def test_vmap_error_axis_size_mismatch():
    """Test that axis_size mismatch raises error."""
    print("=" * 50)
    print("Test: vmap axis_size mismatch error")
    print("=" * 50)
    
    def double(x):
        return x * 2
    
    x = Tensor.ones((5, 3))  # batch of 5
    
    try:
        vmap(double, axis_size=10)(x)  # Mismatch: 10 != 5
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    
    print("✓ vmap axis_size mismatch error works!\n")


def test_vmap_error_all_none_no_axis_size():
    """Test that all in_axes=None without axis_size raises error."""
    print("=" * 50)
    print("Test: vmap all None without axis_size error")
    print("=" * 50)
    
    def double(x):
        return x * 2
    
    x = Tensor.full((3,), 1.0)
    
    try:
        vmap(double, in_axes=None)(x)  # All None, no axis_size
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    
    print("✓ vmap all None error works!\n")


# =============================================================================
# SHAPE-SENSITIVE TESTS (Validate vmap's logical shape handling)
# =============================================================================

def test_vmap_reduce_sum():
    """Test vmap with reduce_sum - validates logical axis interpretation."""
    print("=" * 50)
    print("Test: vmap with reduce_sum (shape-sensitive)")
    print("=" * 50)
    
    from eager import reduce_sum
    
    def row_sum(x):
        # x has logical shape (rows=3, cols=4)
        # axis=0 should reduce ROWS (not batch!)
        return reduce_sum(x, axis=0)
    
    # Input: (batch=5, rows=3, cols=4)
    x = Tensor.ones((5, 3, 4))
    print(f"x.shape: {tuple(x.shape)}")
    
    result = vmap(row_sum)(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    # Should be (batch=5, cols=4) - rows reduced, batch preserved
    assert tuple(result.shape) == (5, 4), f"Expected (5, 4), got {tuple(result.shape)}"
    
    print("✓ vmap reduce_sum works!\n")


def test_vmap_reshape():
    """Test vmap with reshape - validates logical shape handling."""
    print("=" * 50)
    print("Test: vmap with reshape (shape-sensitive)")
    print("=" * 50)
    
    from eager import reshape
    
    def flatten(x):
        # x has logical shape (3, 4) = 12 elements
        # reshape to logical (12,)
        return reshape(x, (12,))
    
    # Input: (batch=5, 3, 4)
    x = Tensor.ones((5, 3, 4))
    print(f"x.shape: {tuple(x.shape)}")
    
    result = vmap(flatten)(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    # Should be (batch=5, 12) - logical flattened, batch preserved
    assert tuple(result.shape) == (5, 12), f"Expected (5, 12), got {tuple(result.shape)}"
    
    print("✓ vmap reshape works!\n")


def test_vmap_nested_reduce_sum():
    """Test nested vmap with reduce_sum - validates multi-level batching."""
    print("=" * 50)
    print("Test: nested vmap with reduce_sum")
    print("=" * 50)
    
    from eager import reduce_sum
    
    def col_sum(x):
        # x has logical shape (cols=4,) after outer vmap
        # This reduces the only logical axis
        return reduce_sum(x, axis=0)
    
    def row_col_sum(x):
        # x has logical shape (rows=3, cols=4)
        # First vmap over rows, then sum each row
        return vmap(col_sum)(x)
    
    # Input: (batch=2, rows=3, cols=4)
    x = Tensor.ones((2, 3, 4))
    print(f"x.shape: {tuple(x.shape)}")
    
    result = vmap(row_col_sum)(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    # Should be (batch=2, rows=3) - cols reduced per row, batch preserved
    assert tuple(result.shape) == (2, 3), f"Expected (2, 3), got {tuple(result.shape)}"
    
    print("✓ nested vmap reduce_sum works!\n")


def test_vmap_reshape_then_reduce():
    """Test vmap with reshape followed by reduce - combined shape ops."""
    print("=" * 50)
    print("Test: vmap reshape then reduce")
    print("=" * 50)
    
    from eager import reduce_sum, reshape
    
    def process(x):
        # x has logical shape (2, 6)
        y = reshape(x, (3, 4))  # reshape to (3, 4)
        return reduce_sum(y, axis=1)  # sum over cols -> (3,)
    
    # Input: (batch=5, 2, 6)
    x = Tensor.ones((5, 2, 6))
    print(f"x.shape: {tuple(x.shape)}")
    
    result = vmap(process)(x)
    
    print(f"result.shape: {tuple(result.shape)}")
    # Should be (batch=5, 3) - reshaped then reduced
    assert tuple(result.shape) == (5, 3), f"Expected (5, 3), got {tuple(result.shape)}"
    
    print("✓ vmap reshape then reduce works!\n")


def main():
    print("\n" + "=" * 60)
    print(" VMAP TRANSFORM TESTS")
    print("=" * 60 + "\n")
    
    # Basic tests
    test_vmap_basic_square()
    test_vmap_binary_same_axes()
    test_vmap_broadcast_none_axis()
    test_vmap_out_axes_nonzero()
    test_vmap_multi_output()
    
    # Decorator tests
    test_vmap_decorator()
    test_vmap_decorator_with_args()
    
    # Advanced tests
    test_vmap_nested()
    test_vmap_triple_nested()
    test_vmap_pytree_inputs()
    test_vmap_same_shape_broadcast()
    test_vmap_negative_axis()
    test_vmap_preserves_batch_dims()
    
    # New feature tests
    test_vmap_axis_size_broadcast()
    test_vmap_dict_in_axes()
    
    # Shape-sensitive tests (validate logical shape handling)
    test_vmap_reduce_sum()
    test_vmap_reshape()
    test_vmap_nested_reduce_sum()
    test_vmap_reshape_then_reduce()
    
    # Error handling tests
    test_vmap_error_inconsistent_batch_sizes()
    test_vmap_error_axis_out_of_bounds()
    test_vmap_error_axis_size_mismatch()
    test_vmap_error_all_none_no_axis_size()
    
    print("=" * 60)
    print(" ALL VMAP TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()


