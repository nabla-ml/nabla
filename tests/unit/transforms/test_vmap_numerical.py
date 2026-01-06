"""Rigorous numerical verification tests for vmap+sharding.

These tests:
1. Create tensors with known values via from_dlpack
2. Apply vmap+sharding operations
3. Evaluate tensors to get actual values
4. Compare against NumPy reference implementations
"""
import pytest
import numpy as np
import asyncio
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec
from nabla.transforms.vmap import vmap
from nabla.ops import unary as unary_ops


def tensor_from_numpy(arr: np.ndarray) -> Tensor:
    """Create Tensor from numpy array."""
    return Tensor.from_dlpack(arr)


def to_numpy(tensor: Tensor) -> np.ndarray:
    """Evaluate tensor and convert to numpy.
    
    For sharded tensors, gathers all shards before conversion.
    """
    # Synchronously realize
    tensor._sync_realize()
    
    # If sharded, we need to gather all shards
    if tensor._impl._storages and len(tensor._impl._storages) > 1:
        sharding = tensor._impl.sharding
        if sharding:
            # Find which dim is sharded
            sharded_dim = None
            for dim_idx, spec in enumerate(sharding.dim_specs):
                if spec.axes:
                    sharded_dim = dim_idx
                    break
            
            # Gather all shards
            shard_arrays = [np.from_dlpack(s) for s in tensor._impl._storages]
            
            if sharded_dim is not None:
                return np.concatenate(shard_arrays, axis=sharded_dim)
            else:
                # All replicated - just return first
                return shard_arrays[0]
        else:
            # No sharding spec but multiple storages - concatenate on axis 0
            return np.concatenate([np.from_dlpack(s) for s in tensor._impl._storages], axis=0)
    
    return np.from_dlpack(tensor)


@pytest.fixture
def mesh():
    return DeviceMesh("test", (4,), ("dp",))


class TestVmapNumericalCorrectness:
    """Numerical verification of vmap operations."""
    
    def test_vmap_mul_numerical(self):
        """Verify vmap multiplication produces correct values."""
        np_data = np.arange(12).reshape(3, 4).astype(np.float32)
        x = tensor_from_numpy(np_data)
        
        @vmap
        def double(row):
            return row * 2
        
        y = double(x)
        
        # Expected: each row multiplied by 2
        expected = np_data * 2
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_vmap_add_numerical(self):
        """Verify vmap addition produces correct values."""
        np_data = np.ones((4, 5), dtype=np.float32) * 3
        x = tensor_from_numpy(np_data)
        
        @vmap
        def add_one(row):
            return row + 1
        
        y = add_one(x)
        
        expected = np_data + 1
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_vmap_compound_ops_numerical(self):
        """Verify vmap with multiple operations."""
        np_data = np.arange(20).reshape(4, 5).astype(np.float32)
        x = tensor_from_numpy(np_data)
        
        @vmap
        def compound(row):
            return (row * 2 + 1) * 3
        
        y = compound(x)
        
        expected = (np_data * 2 + 1) * 3
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestNestedVmapNumerical:
    """Numerical verification of nested vmap."""
    
    def test_double_vmap_numerical(self):
        """Verify double vmap produces correct values."""
        np_data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        x = tensor_from_numpy(np_data)
        
        @vmap
        @vmap
        def square(vec):
            return vec * vec
        
        y = square(x)
        
        expected = np_data * np_data
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_nested_vmap_compound_numerical(self):
        """Verify nested vmap with compound operations."""
        np_data = np.arange(60).reshape(3, 4, 5).astype(np.float32)
        x = tensor_from_numpy(np_data)
        
        @vmap
        @vmap
        def poly(vec):
            return vec * vec + 2 * vec + 1  # (x+1)^2
        
        y = poly(x)
        
        expected = np_data * np_data + 2 * np_data + 1
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestVmapWithShardingNumerical:
    """Numerical verification of vmap combined with sharding."""
    
    def test_vmap_sharded_input_numerical(self, mesh):
        """Verify vmap over sharded input produces correct values."""
        np_data = np.arange(16).reshape(4, 4).astype(np.float32)
        x = tensor_from_numpy(np_data)
        
        # Shard dim 0
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([])])
        
        @vmap
        def double(row):
            return row * 2
        
        y = double(x)
        
        expected = np_data * 2
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_shard_inside_vmap_numerical(self, mesh):
        """Verify sharding inside vmap produces correct values."""
        np_data = np.arange(32).reshape(4, 8).astype(np.float32)
        x = tensor_from_numpy(np_data)
        
        @vmap
        def process(row):
            # Inside vmap, shard the logical row
            sharded = row.shard(mesh, [DimSpec(["dp"])])
            return sharded * 2
        
        y = process(x)
        
        expected = np_data * 2
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestComplexFunctionsNumerical:
    """Test more complex functions with multiple operations."""
    
    def test_mlp_like_function_numerical(self, mesh):
        """Simulate a simple MLP-like computation."""
        # Batch of inputs
        batch_size = 4
        input_dim = 8
        hidden_dim = 16
        
        np_x = np.random.randn(batch_size, input_dim).astype(np.float32)
        np_w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32)
        np_w2 = np.random.randn(hidden_dim, input_dim).astype(np.float32)
        
        x = tensor_from_numpy(np_x)
        w1 = tensor_from_numpy(np_w1)
        w2 = tensor_from_numpy(np_w2)
        
        # Note: We can't use matmul in vmap directly without proper batching support
        # Testing elementwise operations chain instead
        
        @vmap
        def process_row(row):
            # Multiply by constant, add, multiply again
            h = row * 2 + 0.5
            out = h * 3 - 1
            return out
        
        y = process_row(x)
        
        expected = (np_x * 2 + 0.5) * 3 - 1
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)
    
    def test_nested_vmap_elementwise_chain_numerical(self):
        """Test nested vmap with chain of elementwise ops."""
        np_data = np.random.randn(2, 3, 8).astype(np.float32) * 0.5
        x = tensor_from_numpy(np_data)
        
        @vmap
        @vmap
        def chain(vec):
            a = vec * 2
            b = a + 1
            c = b * b  # square
            d = c - 0.5
            return d
        
        y = chain(x)
        
        # NumPy reference
        a = np_data * 2
        b = a + 1
        c = b * b
        expected = c - 0.5
        
        result = to_numpy(y)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)

    def test_vmap_matmul_numerical(self):
        """Verify matmul inside vmap works correctly (batched matvec)."""
        # Batch of matrices (2x3x4) and a vector (4x1)
        np_batch = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        np_vec = np.arange(4).reshape(4, 1).astype(np.float32)
        
        batch = tensor_from_numpy(np_batch)
        vec = tensor_from_numpy(np_vec)
        
        # vmap over batch dim 0 of first arg, broadcast second arg
        @vmap(in_axes=(0, None))
        def batched_matvec(mat, v):
            return mat @ v
            
        y = batched_matvec(batch, vec)
        
        # NumPy validation (einsum is clearer than broadcasting for verification)
        expected = np.einsum('bij,jk->bik', np_batch, np_vec)
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_vmap_unary_ops_numerical(self):
        """Verify unary operations (neg, abs, etc) work inside vmap."""
        np_data = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        x = tensor_from_numpy(np_data)
        
        @vmap
        def unary_chain(row):
            # Test negation and abs
            a = -row
            b = abs(a)
            return b
            
        y = unary_chain(x)
        
        expected = np.abs(-np_data)
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_vmap_reduction_numerical(self):
        """Verify vmap with reduction (sum) works correctly."""
        # (batch=2, rows=4, cols=5)
        np_data = np.arange(40).reshape(2, 4, 5).astype(np.float32)
        x = tensor_from_numpy(np_data)
        
        @vmap
        def reduce_dim0(row):
            # Sum logical axis 0 (physical axis 1) -> result shape (5,)
            # Implementation must shift axis=0 to axis=1
            return row.sum(axis=0)
            
        y = reduce_dim0(x)
        
        # Expected: sum over axis 1 of original data -> (2, 5)
        expected = np_data.sum(axis=1)
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)


    def test_vmap_pure_broadcast_numerical(self):
        """Verify vmap with axis_size (pure broadcast)."""
        x = Tensor.constant(2.0)
        
        # Create 5 copies of x+1
        @vmap(in_axes=None, axis_size=5)
        def create_batch(val):
            return val + 1.0
            
        y = create_batch(x)
        
        expected = np.full((5,), 3.0, dtype=np.float32)
        result = to_numpy(y)
        
        np.testing.assert_allclose(result, expected)
