# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Tests for vmap with spmd_axis_name parameter.

This file tests the CRITICAL composability question:
    Does spmd_axis_name (batch dim sharding) compose correctly with
    logical dim sharding inside the vmapped function?

Test hierarchy:
1. TestSpmdAxisNameBasic - vmap with spmd_axis_name alone
2. TestSpmdAxisWithLogicalSharding - THE CRITICAL COMPOSABILITY TESTS
3. TestSpmdAxisNestedVmap - Nested vmap with spmd_axis_name
"""

import pytest
import numpy as np

import nabla
from nabla import DeviceMesh, P, vmap
from nabla import add, mul, relu, sigmoid, matmul, reduce_sum, mean
from nabla.core import trace

from tests.conftest import (
    make_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_physical_shape, assert_batch_dims,
    assert_is_sharded,
)


# =============================================================================
# Basic spmd_axis_name Tests 
# =============================================================================

class TestSpmdAxisNameBasic:
    """Test vmap with spmd_axis_name parameter alone (no logical sharding)."""
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes,spmd_axis", [
        ((2,), ("dp",), "dp"),
        ((4,), ("dp",), "dp"),
        ((2, 4), ("dp", "tp"), "dp"),
    ])
    def test_spmd_basic_relu(self, mesh_shape, mesh_axes, spmd_axis):
        """vmap with spmd_axis_name on unary op."""
        batch = 8
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)
        
        # Create input and shard it on the batch dimension manually
        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Define the vmapped function for tracing
        vmapped_relu = vmap(relu, spmd_axis_name=spmd_axis)
        
        # === TRACE DEBUG: See what operations are generated ===
        print(f"\n{'='*60}")
        print(f"TEST: test_spmd_basic_relu")
        print(f"mesh_shape={mesh_shape}, mesh_axes={mesh_axes}, spmd_axis={spmd_axis}")
        print(f"{'='*60}")
        t = trace(vmapped_relu, x)
        print(t)
        print(f"{'='*60}\n")
        # === END TRACE DEBUG ===
        
        # Apply vmap with spmd_axis_name
        result = vmapped_relu(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)
        
        # Verify batch dimension is sharded on spmd_axis
        # Note: after vmap, batch_dims = 0, so physical shape = logical shape
        if result.sharding:
            spec = result.sharding
            # The first physical dimension should be sharded on spmd_axis
            assert spmd_axis in spec.dim_specs[0].axes, \
                f"Expected batch dim sharded on {spmd_axis}, got {spec.dim_specs[0]}"
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes,spmd_axis", [
        ((2,), ("dp",), "dp"),
        ((2, 4), ("dp", "tp"), "dp"),
    ])
    def test_spmd_basic_add(self, mesh_shape, mesh_axes, spmd_axis):
        """vmap with spmd_axis_name on binary op."""
        batch = 8
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)
        
        np_a = make_array(batch, 16, seed=42)
        np_b = make_array(batch, 16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        result = vmap(add, spmd_axis_name=spmd_axis)(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)
        
        # Verify sharding
        if result.sharding:
            assert spmd_axis in result.sharding.dim_specs[0].axes
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes,spmd_axis", [
        ((2,), ("dp",), "dp"),
        ((2, 4), ("dp", "tp"), "dp"),
    ])
    def test_spmd_basic_matmul(self, mesh_shape, mesh_axes, spmd_axis):
        """vmap with spmd_axis_name on matmul."""
        batch = 4
        M, K, N = 8, 16, 12
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)
        
        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)
        
        result = vmap(matmul, in_axes=(0, None), spmd_axis_name=spmd_axis)(x, w)
        expected = np_x @ np_w
        
        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)
        
        # First dim should be sharded on spmd_axis
        if result.sharding:
            assert spmd_axis in result.sharding.dim_specs[0].axes
    
    def test_spmd_asymmetric_mesh(self, mesh_2x4):
        """vmap with spmd_axis_name on asymmetric mesh."""
        batch = 8
        
        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        result = vmap(relu, spmd_axis_name="dp")(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)
        
        # Verify sharded on dp
        if result.sharding:
            assert "dp" in result.sharding.dim_specs[0].axes


# =============================================================================
# THE CRITICAL TESTS: spmd_axis_name + Logical Sharding Composability
# =============================================================================

class TestSpmdAxisWithLogicalSharding:
    """Test the critical composability: spmd_axis_name + logical dim sharding.
    
    This is THE KEY question: when we use spmd_axis_name to shard the batch dim,
    AND we shard logical dims inside the vmapped function, do they compose correctly?
    
    Expected behavior:
    - Physical shape has batch dim at position 0
    - Batch dim sharded on spmd_axis_name
    - Logical dims sharded per user annotation
    - Result: multi-axis sharding on physical tensor
    """
    
    def test_spmd_plus_logical_sharding_relu(self, mesh_2x4):
        """Batch dim sharded on dp, logical dim on tp."""
        batch = 8
        mesh = mesh_2x4  # (2, 4) with ("dp", "tp")
        
        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        @vmap(spmd_axis_name="dp", mesh=mesh)  # Shard batch on dp
        def f(row):  # row: (16,) logical
            row_sharded = row.shard(mesh, P("tp"))  # Shard logical dim on tp
            return relu(row_sharded)
        
        # === TRACE DEBUG: Critical composability test ===
        print(f"\n{'='*60}")
        print(f"TEST: test_spmd_plus_logical_sharding_relu")
        print(f"CRITICAL: batch dim on 'dp' + logical dim on 'tp'")
        print(f"mesh_shape=(2, 4), mesh_axes=('dp', 'tp')")
        print(f"Expected: multi-axis sharding [dp, tp] on output")
        print(f"{'='*60}")
        t = trace(f, x)
        print(t)
        print(f"{'='*60}\n")
        # === END TRACE DEBUG ===
        
        result = f(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)
        
        # CRITICAL VERIFICATION: Both axes should be sharded
        spec = result.sharding
        assert spec is not None, "Result should have sharding"
        
        # Physical dim 0 (batch) sharded on dp
        assert "dp" in spec.dim_specs[0].axes, \
            f"Batch dim should be sharded on dp, got {spec.dim_specs[0]}"
        
        # Physical dim 1 (features) sharded on tp
        assert "tp" in spec.dim_specs[1].axes, \
            f"Features dim should be sharded on tp, got {spec.dim_specs[1]}"
    
    def test_spmd_plus_logical_sharding_add(self, mesh_2x4):
        """Binary op with batch sharded on dp, logical on tp."""
        batch = 8
        mesh = mesh_2x4
        
        np_a = make_array(batch, 16, seed=42)
        np_b = make_array(batch, 16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        @vmap(spmd_axis_name="dp", mesh=mesh)
        def f(row_a, row_b):
            # Shard both on tp
            row_a_s = row_a.shard(mesh, P("tp"))
            row_b_s = row_b.shard(mesh, P("tp"))
            return add(row_a_s, row_b_s)
        
        result = f(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)
        
        # Verify both axes
        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[1].axes
    
    def test_spmd_plus_logical_matmul_column_parallel(self, mesh_2x4):
        """Matmul with batch sharded, weight column-parallel."""
        batch = 4
        M, K, N = 8, 16, 32
        mesh = mesh_2x4
        
        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)
        
        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)
        
        @vmap(in_axes=(0, None), spmd_axis_name="dp", mesh=mesh)
        def f(xi, wi):  # xi: (M, K), wi: (K, N)
            # Column parallel: shard weight on N axis (tp)
            wi_sharded = wi.shard(mesh, P(None, "tp"))
            return matmul(xi, wi_sharded)
        
        result = f(x, w)
        expected = np_x @ np_w
        
        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)
        
        # Verify: batch on dp, output features on tp
        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes  # Batch
        assert "tp" in spec.dim_specs[2].axes  # Output features (N)
    
    def test_spmd_plus_logical_matmul_row_parallel(self, mesh_2x4):
        """Matmul with batch sharded, weight row-parallel."""
        batch = 4
        M, K, N = 8, 32, 16
        mesh = mesh_2x4
        
        np_x = make_array(batch, M, K, seed=42)
        np_w = make_array(K, N, seed=43)
        
        x = tensor_from_numpy(np_x)
        w = tensor_from_numpy(np_w)
        
        @vmap(in_axes=(0, None), spmd_axis_name="dp", mesh=mesh)
        def f(xi, wi):  # xi: (M, K), wi: (K, N)
            # Row parallel: shard both on K axis
            xi_sharded = xi.shard(mesh, P(None, "tp"))
            wi_sharded = wi.shard(mesh, P("tp", None))
            return matmul(xi_sharded, wi_sharded)
        
        # === TRACE DEBUG: Row parallel matmul requires AllReduce! ===
        print(f"\n{'='*60}")
        print(f"TEST: test_spmd_plus_logical_matmul_row_parallel")
        print(f"CRITICAL: Contracting dim K is sharded on 'tp'")
        print(f"Expected: AllReduce should appear in trace!")
        print(f"mesh_shape=(2, 4), mesh_axes=('dp', 'tp')")
        print(f"{'='*60}")
        t = trace(f, x, w)
        print(t)
        # Check if all_reduce appears in trace
        trace_str = str(t)
        if "all_reduce" in trace_str.lower():
            print("✅ AllReduce detected in trace - CORRECT!")
        else:
            print("❌ WARNING: No AllReduce detected - may be INCORRECT!")
        print(f"{'='*60}\n")
        # === END TRACE DEBUG ===
        
        result = f(x, w)
        expected = np_x @ np_w
        
        assert_shape(result, (batch, M, N))
        assert_allclose(result, expected, rtol=1e-4)
        
        # Verify batch sharded on dp
        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
    
    def test_spmd_plus_logical_reduction(self, mesh_2x4):
        """Reduction op with batch and logical sharding."""
        batch = 8
        mesh = mesh_2x4
        
        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        @vmap(spmd_axis_name="dp", mesh=mesh)
        def f(row):  # row: (16,)
            row_sharded = row.shard(mesh, P("tp"))
            return reduce_sum(row_sharded, axis=0)  # Sum over features
        
        # === TRACE DEBUG: Reduction over sharded axis requires AllReduce! ===
        print(f"\n{'='*60}")
        print(f"TEST: test_spmd_plus_logical_reduction")
        print(f"CRITICAL: Reducing sharded axis (tp)")
        print(f"Expected: AllReduce should appear after local reduce_sum!")
        print(f"mesh_shape=(2, 4), mesh_axes=('dp', 'tp')")
        print(f"{'='*60}")
        t = trace(f, x)
        print(t)
        trace_str = str(t)
        if "all_reduce" in trace_str.lower():
            print("✅ AllReduce detected - CORRECT for reducing sharded axis!")
        else:
            print("❌ WARNING: No AllReduce detected - reduction may be INCORRECT!")
        print(f"{'='*60}\n")
        # === END TRACE DEBUG ===
        
        result = f(x)
        expected = np.sum(np_x, axis=1)
        
        assert_shape(result, (batch,))
        assert_allclose(result, expected)
        
        # Result is 1D, batch dim should be sharded on dp
        spec = result.sharding
        if spec:
            assert "dp" in spec.dim_specs[0].axes
    
    def test_spmd_plus_logical_3d_mesh(self, mesh_3d_2x4x2):
        """With 3D mesh: batch on dp, features on tp."""
        batch = 8
        mesh = mesh_3d_2x4x2  # (2, 4, 2) with ("dp", "tp", "pp")
        
        np_x = make_array(batch, 16, seed=42)
        x = tensor_from_numpy(np_x)
        
        @vmap(spmd_axis_name="dp", mesh=mesh)
        def f(row):
            row_sharded = row.shard(mesh, P("tp"))
            return relu(row_sharded)
        
        result = f(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 16))
        assert_allclose(result, expected)
        
        # Verify sharding on both dp and tp
        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes
        assert "tp" in spec.dim_specs[1].axes


# =============================================================================
# Nested vmap with spmd_axis_name
# =============================================================================

class TestSpmdAxisNestedVmap:
    """Test nested vmap with spmd_axis_name."""
    
    def test_nested_vmap_spmd_outer(self, mesh_2x4):
        """Nested vmap: spmd_axis_name on outer vmap only."""
        outer_batch = 4
        inner_batch = 6
        features = 8
        mesh = mesh_2x4
        
        np_x = make_array(outer_batch, inner_batch, features, seed=42)
        x = tensor_from_numpy(np_x)
        
        # Outer vmap with spmd_axis_name
        @vmap(spmd_axis_name="dp", mesh=mesh)
        def outer(batch_x):
            # Inner vmap without spmd_axis_name
            return vmap(relu)(batch_x)
        
        result = outer(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (outer_batch, inner_batch, features))
        assert_allclose(result, expected)
        
        # Outer batch should be sharded on dp
        spec = result.sharding
        if spec:
            assert "dp" in spec.dim_specs[0].axes
    
    def test_nested_vmap_spmd_with_logical(self, mesh_2x4):
        """Nested vmap: spmd_axis_name + logical sharding."""
        outer_batch = 4
        inner_batch = 6
        features = 16
        mesh = mesh_2x4
        
        np_x = make_array(outer_batch, inner_batch, features, seed=42)
        x = tensor_from_numpy(np_x)
        
        @vmap(spmd_axis_name="dp", mesh=mesh)  # Outer batch on dp
        def outer(batch_x):  # batch_x: (inner_batch, features)
            @vmap
            def inner(row):  # row: (features,)
                row_sharded = row.shard(mesh, P("tp"))  # Features on tp
                return relu(row_sharded)
            return inner(batch_x)
        
        result = outer(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (outer_batch, inner_batch, features))
        assert_allclose(result, expected)
        
        # Verify both shardings
        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes  # Outer batch
        assert "tp" in spec.dim_specs[2].axes  # Features
    
    def test_nested_vmap_both_spmd(self, mesh_3d_2x4x2):
        """Nested vmap: both with spmd_axis_name on different axes."""
        outer_batch = 4
        inner_batch = 8
        features = 16
        mesh = mesh_3d_2x4x2  # (2, 4, 2)
        
        np_x = make_array(outer_batch, inner_batch, features, seed=42)
        x = tensor_from_numpy(np_x)
        
        @vmap(spmd_axis_name="dp", mesh=mesh)  # Outer on dp
        def outer(batch_x):
            @vmap(spmd_axis_name="pp", mesh=mesh)  # Inner on pp
            def inner(row):
                row_sharded = row.shard(mesh, P("tp"))  # Features on tp
                return relu(row_sharded)
            return inner(batch_x)
        
        result = outer(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (outer_batch, inner_batch, features))
        assert_allclose(result, expected)
        
        # All three axes should be sharded
        spec = result.sharding
        assert "dp" in spec.dim_specs[0].axes  # Outer batch
        assert "pp" in spec.dim_specs[1].axes  # Inner batch
        assert "tp" in spec.dim_specs[2].axes  # Features


__all__ = [
    "TestSpmdAxisNameBasic",
    "TestSpmdAxisWithLogicalSharding",
    "TestSpmdAxisNestedVmap",
]
