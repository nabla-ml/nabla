# ===----------------------------------------------------------------------=== #
# Nabla 2026
# ===----------------------------------------------------------------------=== #

"""Comprehensive numerical verification tests for vmap + sharding.

This file rigorously tests the interaction between vmap and sharding across:
1. View Operations (Reshape, Transpose, Squeeze, Broadcast)
2. Matrix Multiplication (Data Parallel, Model Parallel, Fully Sharded)
3. Explicit SPMD Axis Mapping (spmd_axis_name)
4. Nested Compositions
"""

import pytest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.transforms.vmap import vmap
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla import reshape, swap_axes, squeeze, unsqueeze, broadcast_to

# Helper for numerical comparison
def assert_allclose(t: Tensor, expected: np.ndarray, rtol=1e-5):
    t._sync_realize()
    # Handle sharded gather automatically via to_numpy() if implemented, 
    # else manual gather. Assuming to_numpy() handles it from previous context.
    result = t.to_numpy()
    np.testing.assert_allclose(result, expected, rtol=rtol, err_msg=f"Shape mismatch: {result.shape} vs {expected.shape}")


@pytest.fixture
def mesh():
    return DeviceMesh("test_mesh", (4,), ("dp",))

@pytest.fixture
def mesh_2d():
    return DeviceMesh("mesh_2d", (2, 2), ("x", "y"))


class TestVmapViewOpsSharding:
    """Test view operations inside vmap with sharded inputs."""

    def test_vmap_reshape_sharded(self, mesh):
        """vmap(reshape): (B, 8) -> (B, 2, 4) with sharded B."""
        # Input: (4, 8) sharded on "dp" (dim 0)
        np_x = np.arange(32).reshape(4, 8).astype(np.float32)
        x = Tensor.from_dlpack(np_x)
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([])])

        @vmap
        def reshape_row(row):
            # row: (8,) -> (2, 4)
            return reshape(row, (2, 4))

        y = reshape_row(x)
        
        # Check sharding: Batch dim should still be sharded on "dp"
        # Physical shape: (4, 2, 4)
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"] 
        assert spec.dim_specs[1].is_replicated()
        assert spec.dim_specs[2].is_replicated()

        # Check values
        expected = np_x.reshape(4, 2, 4)
        assert_allclose(y, expected)

    def test_vmap_transpose_sharded(self, mesh):
        """vmap(transpose): (B, M, N) -> (B, N, M) with sharded B."""
        # Input: (4, 2, 3) sharded on "dp" (dim 0)
        np_x = np.arange(24).reshape(4, 2, 3).astype(np.float32)
        x = Tensor.from_dlpack(np_x)
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])

        @vmap
        def transpose_mat(mat):
            # mat: (2, 3) -> (3, 2)
            return swap_axes(mat, 0, 1)

        y = transpose_mat(x)

        # Check sharding
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        
        # Check values
        expected = np.transpose(np_x, (0, 2, 1))
        assert_allclose(y, expected)

    def test_vmap_broadcast_sharded(self, mesh):
        """vmap(broadcast): (B, 1) -> (B, 4) with sharded B."""
        # Input: (4, 1) sharded on "dp" (dim 0)
        np_x = np.arange(4).reshape(4, 1).astype(np.float32)
        x = Tensor.from_dlpack(np_x)
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([])])

        @vmap
        def broadcast_row(row):
            # row: (1,) -> (4,)
            return broadcast_to(row, (4,))

        y = broadcast_row(x)

        # Check sharding
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]

        # Check values
        expected = np.broadcast_to(np_x, (4, 4))
        assert_allclose(y, expected)


    def test_vmap_unsqueeze_sharded(self, mesh):
        """vmap(unsqueeze): (B, N) -> (B, 1, N) with sharded B."""
        # Input: (4, 8) sharded on "dp" (dim 0)
        np_x = np.arange(32).reshape(4, 8).astype(np.float32)
        x = Tensor.from_dlpack(np_x)
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([])])

        @vmap
        def add_dim(row):
            # row: (8,) -> (1, 8)
            return unsqueeze(row, axis=0)

        y = add_dim(x)
        
        # Check sharding: Batch dim (0) should be "dp"
        # Dim 1 is new (1), Dim 2 is old (8)
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        assert spec.dim_specs[1].is_replicated()
        assert spec.dim_specs[2].is_replicated()

        expected = np.expand_dims(np_x, axis=1)
        assert_allclose(y, expected)

    def test_vmap_squeeze_sharded(self, mesh):
        """vmap(squeeze): (B, 1, N) -> (B, N) with sharded B."""
        # Input: (4, 1, 8) sharded on "dp" (dim 0)
        np_x = np.arange(32).reshape(4, 1, 8).astype(np.float32)
        x = Tensor.from_dlpack(np_x)
        x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])

        @vmap
        def remove_dim(row):
            # row: (1, 8) -> (8,)
            return squeeze(row, axis=0)

        y = remove_dim(x)

        # Check sharding
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        assert spec.dim_specs[1].is_replicated()

        expected = np.squeeze(np_x, axis=1)
        assert_allclose(y, expected)


class TestVmapMathOpsSharding:
    """Test mathematical operations (Binary, Reduction) inside vmap with sharded inputs."""

    def test_vmap_binary_ops_sharded(self, mesh):
        """vmap(x + y): x sharded, y replicated."""
        # x: (4, 8) [DP]
        # y: (4, 8) [R] (broadcasted or passed in)
        # Let's use simple scalars first then full tensors
        np_x = np.arange(32).reshape(4, 8).astype(np.float32)
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["dp"]), DimSpec([])])
        y = Tensor.full((4, 8), 10.0) # Unsharded

        @vmap(in_axes=(0, 0))
        def add_op(a, b):
            return a + b

        result = add_op(x, y)
        
        # Output sharding: Should inherit DP from x
        spec = result._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        
        expected = np_x + 10.0
        assert_allclose(result, expected)

    def test_vmap_reduction_sharded(self, mesh):
        """vmap(sum(x, axis=0)): Reduction inside vmap over NON-batch dim."""
        # x: (4, 8) [DP]
        np_x = np.arange(32).reshape(4, 8).astype(np.float32)
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["dp"]), DimSpec([])])

        @vmap
        def sum_row(row):
            # row: (8,)
            return row.sum()

        result = sum_row(x) # (4,)
        
        # Output sharding: (4,) sharded on DP
        spec = result._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        
        expected = np.sum(np_x, axis=1)
        assert_allclose(result, expected)

    def test_vmap_reduction_sharded_keepdims(self, mesh):
        """vmap(sum(x, keepdims=True))."""
        np_x = np.arange(32).reshape(4, 8).astype(np.float32)
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["dp"]), DimSpec([])])

        @vmap
        def sum_row_kd(row):
            return row.sum(keepdims=True)

        result = sum_row_kd(x) # (4, 1)
        
        spec = result._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        assert spec.dim_specs[1].is_replicated()
        
        expected = np.sum(np_x, axis=1, keepdims=True)
        assert_allclose(result, expected)


class TestVmapMatmulSharding:
    """Test matrix multiplication inside vmap with various sharding strategies."""

    def test_data_parallel_matmul(self, mesh):
        """Data Parallel: vmap(x @ w) where x is sharded on batch dim."""
        # x: (4, 8) sharded on "dp"
        # w: (8, 4) replicated
        np.random.seed(42)  # Fix seed for determinism
        np_x = np.random.randn(4, 8).astype(np.float32)
        np_w = np.random.randn(8, 4).astype(np.float32)
        
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["dp"]), DimSpec([])])
        w = Tensor.from_dlpack(np_w) # Replicated by default

        @vmap(in_axes=(0, None))
        def linear(x_row, weight):
            return x_row @ weight

        y = linear(x, w)

        # Output should be (4, 4) sharded on "dp"
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        assert spec.dim_specs[1].is_replicated()

        expected = np_x @ np_w
        assert_allclose(y, expected)

    def test_model_parallel_weights(self, mesh):
        """Model Parallel: vmap(x @ w) where w is sharded."""
        # x: (4, 8) replciated (batch)
        # w: (8, 4) sharded on "dp" (dim 1 - output features)
        np_x = np.random.randn(4, 8).astype(np.float32)
        np_w = np.random.randn(8, 4).astype(np.float32)

        x = Tensor.from_dlpack(np_x)
        # Weights sharded on output columns -> Column/Tensor Parallelism
        w = Tensor.from_dlpack(np_w).shard(mesh, [DimSpec([]), DimSpec(["dp"])])

        @vmap(in_axes=(0, None))
        def linear(x_row, weight):
            return x_row @ weight

        y = linear(x, w)

        # Output (4, 4) should be sharded on dim 1 ("dp") because weights were
        # Batch dim (0) effectively replicated (since input was)
        spec = y._impl.sharding
        assert spec.dim_specs[0].is_replicated()
        assert spec.dim_specs[1].axes == ["dp"]

        expected = np_x @ np_w
        assert_allclose(y, expected)

    def test_sharded_contracting_dim(self, mesh):
        """AllReduce Case: contracting dimension is sharded."""
        # x: (4, 8) 
        # w: (8, 4)
        # Shard the contracting dim '8' (dim 1 of x, dim 0 of w) on "dp"
        np_x = np.random.randn(4, 8).astype(np.float32)
        np_w = np.random.randn(8, 4).astype(np.float32)

        # We need manual sharding setup to test this specific interaction
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec([]), DimSpec(["dp"])])
        w = Tensor.from_dlpack(np_w).shard(mesh, [DimSpec(["dp"]), DimSpec([])])
        
        # Problem: vmap(x) splits x on dim 0. If we want contracting dim sharded inside vmap:
        # vmap extracts row: (8,). Row should be sharded on "dp" (dim 0 of row)
        # Weight is (8, 4). Sharded on "dp" (dim 0)
        # row @ w -> (4,)
        # (8,){dp} @ (8, 4){dp, R} -> Contracting dim match -> AllReduce -> (4,)

        @vmap(in_axes=(0, None))
        def linear(x_row, weight):
            # x_row is (8,) sharded on "dp" (inherited from x dim 1)
            return x_row @ weight

        y = linear(x, w)

        # Result should be replicated (AllReduce happened)
        # Because we reduced over specific axis "dp" which was the only sharded axis
        spec = y._impl.sharding
        assert spec.dim_specs[0].is_replicated() # Batch
        assert spec.dim_specs[1].is_replicated() # Output features

        expected = np_x @ np_w
        assert_allclose(y, expected)


class TestSpmdAxisName:
    """Test explicit spmd_axis_name integration."""

    def test_spmd_axis_name_correctness(self, mesh):
        """Verify spmd_axis_name produces correct numerical output."""
        np_input = np.arange(16).reshape(4, 4).astype(np.float32)
        x = Tensor.from_dlpack(np_input)
        
        # Start fully replicated
        x = x.shard(mesh, [DimSpec([]), DimSpec([])])
        
        @vmap(spmd_axis_name="dp")
        def double(row):
            # Inside: row should be local slice
            # If sharding works, this runs on (1, 4) slices (since 4 devices, batch 4)
            return row * 2

        y = double(x)

        # Output should be sharded on "dp"
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]

        expected = np_input * 2
        assert_allclose(y, expected)

    def test_spmd_axis_name_matmul(self, mesh):
        """Verify spmd_axis_name with matmul."""
        # Batch 4, features 8
        np_x = np.random.randn(4, 8).astype(np.float32)
        np_w = np.random.randn(8, 4).astype(np.float32)

        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec([]), DimSpec([])])
        w = Tensor.from_dlpack(np_w)

        @vmap(in_axes=(0, None), spmd_axis_name="dp")
        def linear(row, weight):
            return row @ weight

        y = linear(x, w)

        # Expected: Result (4, 4) sharded on "dp" (dim 0)
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        
        expected = np_x @ np_w
        assert_allclose(y, expected)


class TestNestedVmapSharding:
    """Test nested vmap compositions with sharding."""

    def test_nested_vmap_sharded_input(self, mesh):
        """vmap(vmap(op)) over sharded input."""
        # (Batch1=4, Batch2=2, Feat=8)
        # Shard Batch1 on "dp"
        np_x = np.random.randn(4, 2, 8).astype(np.float32)
        x = Tensor.from_dlpack(np_x).shard(mesh, [DimSpec(["dp"]), DimSpec([]), DimSpec([])])

        @vmap 
        @vmap
        def double(val):
            return val * 2

        y = double(x)

        # Spec should verify Batch1 is still sharded
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["dp"]
        assert spec.dim_specs[1].is_replicated()

        assert_allclose(y, np_x * 2)

    def test_nested_spmd_axis(self, mesh_2d):
        """vmap with spmd_axis_name nested."""
        # Mesh (2, 2) x, y
        # Batch1=2 (map to x), Batch2=2 (map to y), Feat=4
        np_x = np.arange(16).reshape(2, 2, 4).astype(np.float32)
        
        # Start replicated
        x = Tensor.from_dlpack(np_x).shard(mesh_2d, [DimSpec([]), DimSpec([]), DimSpec([])])

        # Map outer vmap to "x", inner vmap to "y"
        @vmap(spmd_axis_name="x")
        @vmap(spmd_axis_name="y") 
        def add_one(row):
            return row + 1

        y = add_one(x)

        # Check resulting sharding: [x, y, R]
        spec = y._impl.sharding
        assert spec.dim_specs[0].axes == ["x"]
        assert spec.dim_specs[1].axes == ["y"]
        assert spec.dim_specs[2].is_replicated()

        assert_allclose(y, np_x + 1)
