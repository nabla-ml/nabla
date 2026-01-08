# ===----------------------------------------------------------------------=== #
# Nabla 2025
# ===----------------------------------------------------------------------=== #
"""Tests for sharding propagation in binary operations.

This file specifically tests that binary operations:
1. Properly propagate sharding when both inputs have the same sharding
2. Handle mismatched sharding scenarios (implicit resharding)
3. Work correctly with broadcast semantics + sharding

These tests verify the elementwise_template and broadcast_template
are correctly applied in binary operations.
"""

import pytest
import numpy as np

import nabla
from nabla import DeviceMesh, P
from nabla import add, sub, mul, div, matmul
from nabla.sharding.spec import DimSpec, ShardingSpec

from tests.conftest import (
    make_array, make_positive_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_is_sharded,
    shard_on_axis, shard_on_axes, replicated,
)


# =============================================================================
# Binary Ops - Same Sharding (Elementwise Propagation)
# =============================================================================

class TestBinaryOpsSameSharding:
    """Test binary ops where both inputs have identical sharding."""
    
    @pytest.mark.parametrize("op,numpy_op", [
        (add, lambda a, b: a + b),
        (sub, lambda a, b: a - b),
        (mul, lambda a, b: a * b),
    ])
    @pytest.mark.parametrize("mesh_shape,mesh_axes", [
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        ((2, 2), ("dp", "tp")),
    ])
    def test_binary_same_sharding_axis0(self, op, numpy_op, mesh_shape, mesh_axes):
        """Binary op with both inputs sharded on axis 0."""
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)
        
        np_a = make_array(8, 16, seed=42)
        np_b = make_array(8, 16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # Shard both on axis 0
        a_sharded = a.shard(mesh, P(mesh_axes[0], None))
        b_sharded = b.shard(mesh, P(mesh_axes[0], None))
        
        result = op(a_sharded, b_sharded)
        expected = numpy_op(np_a, np_b)
        
        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("op,numpy_op", [
        (add, lambda a, b: a + b),
        (mul, lambda a, b: a * b),
    ])
    @pytest.mark.parametrize("mesh_shape,mesh_axes", [
        ((2,), ("tp",)),
        ((4,), ("tp",)),
    ])
    def test_binary_same_sharding_axis1(self, op, numpy_op, mesh_shape, mesh_axes):
        """Binary op with both inputs sharded on axis 1 (tensor parallel)."""
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)
        
        np_a = make_array(8, 16, seed=42)
        np_b = make_array(8, 16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # Shard both on axis 1 (tensor parallel axis)
        a_sharded = a.shard(mesh, P(None, mesh_axes[0]))
        b_sharded = b.shard(mesh, P(None, mesh_axes[0]))
        
        result = op(a_sharded, b_sharded)
        expected = numpy_op(np_a, np_b)
        
        assert_shape(result, (8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)
    
    def test_div_same_sharding(self, mesh_1d):
        """Div with both inputs sharded identically."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_positive_array(8, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)
        
        result = div(a, b)
        expected = np_a / np_b
        
        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


# =============================================================================
# Binary Ops - Sharded + Replicated
# =============================================================================

class TestBinaryOpsShardedReplicated:
    """Test binary ops where one input is sharded and one is replicated."""
    
    @pytest.mark.parametrize("op,numpy_op", [
        (add, lambda a, b: a + b),
        (mul, lambda a, b: a * b),
    ])
    def test_sharded_plus_replicated(self, op, numpy_op, mesh_1d):
        """Sharded input + replicated input."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = replicated(tensor_from_numpy(np_b), mesh_1d)
        
        result = op(a, b)
        expected = numpy_op(np_a, np_b)
        
        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)  # Result should inherit sharding from a
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("op,numpy_op", [
        (add, lambda a, b: a + b),
        (mul, lambda a, b: a * b),
    ])
    def test_replicated_plus_sharded(self, op, numpy_op, mesh_1d):
        """Replicated input + sharded input (order flipped)."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)
        
        a = replicated(tensor_from_numpy(np_a), mesh_1d)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)
        
        result = op(a, b)
        expected = numpy_op(np_a, np_b)
        
        assert_shape(result, (8, 4))
        assert_is_sharded(result, True)  # Result should inherit sharding from b
        assert_allclose(result, expected)
    
    def test_sharded_plus_unsharded(self, mesh_1d):
        """Sharded input + completely unsharded (no mesh) input."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = tensor_from_numpy(np_b)  # No sharding at all
        
        result = add(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (8, 4))
        assert_allclose(result, expected)


# =============================================================================
# Binary Ops - Mismatched Sharding (Requires Resharding)
# =============================================================================

class TestBinaryOpsMismatchedSharding:
    """Test binary ops where inputs have different sharding - requires implicit resharding."""
    
    def test_different_sharded_axes(self, mesh_2d):
        """Inputs sharded on different axes - should work via resharding."""
        np_a = make_array(8, 8, seed=42)
        np_b = make_array(8, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # a sharded on axis 0 (dp), b sharded on axis 1 (tp)
        a_sharded = a.shard(mesh_2d, P("dp", None))
        b_sharded = b.shard(mesh_2d, P(None, "tp"))
        
        result = add(a_sharded, b_sharded)
        expected = np_a + np_b
        
        assert_shape(result, (8, 8))
        assert_allclose(result, expected)
    
    def test_different_mesh_axes_same_dim(self):
        """Inputs sharded on same tensor dim but different mesh axes."""
        mesh = DeviceMesh("mesh", (2, 2), ("dp", "tp"))
        
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # Both sharded on axis 0, but using different mesh axes
        a_sharded = a.shard(mesh, P("dp", None))
        b_sharded = b.shard(mesh, P("tp", None))
        
        result = add(a_sharded, b_sharded)
        expected = np_a + np_b
        
        assert_shape(result, (8, 4))
        assert_allclose(result, expected)


# =============================================================================
# Binary Ops - Broadcasting + Sharding
# =============================================================================

class TestBinaryOpsBroadcastSharding:
    """Test binary ops with broadcasting and sharding combined."""
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", [
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        ((2, 2), ("dp", "tp")),
    ])
    def test_broadcast_sharded_with_vector(self, mesh_shape, mesh_axes):
        """Broadcast: (8, 16) sharded + (16,) -> (8, 16) sharded."""
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)
        
        np_a = make_array(8, 16, seed=42)
        np_b = make_array(16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # Shard a on first axis
        a_sharded = a.shard(mesh, P(mesh_axes[0], None))
        
        result = add(a_sharded, b)
        expected = np_a + np_b
        
        assert_shape(result, (8, 16))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", [
        ((2,), ("tp",)),
        ((4,), ("tp",)),
    ])
    def test_broadcast_with_sharded_vector(self, mesh_shape, mesh_axes):
        """Broadcast: (8, 16) + (16,) sharded -> proper broadcast with sharding."""
        mesh = DeviceMesh("mesh", mesh_shape, mesh_axes)
        
        np_a = make_array(8, 16, seed=42)
        np_b = make_array(16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # Shard b (the vector) - this is like sharding a bias term
        b_sharded = b.shard(mesh, P(mesh_axes[0]))
        
        result = add(a, b_sharded)
        expected = np_a + np_b
        
        assert_shape(result, (8, 16))
        assert_allclose(result, expected)
    
    def test_broadcast_both_sharded_compatible(self):
        """Both inputs sharded on compatible dims during broadcast."""
        mesh = DeviceMesh("mesh", (2,), ("tp",))
        
        np_a = make_array(8, 16, seed=42)  # (8, 16)
        np_b = make_array(16, seed=43)      # (16,)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # Shard a on axis 1, b on axis 0 (after broadcast alignment, they match)
        a_sharded = a.shard(mesh, P(None, "tp"))
        b_sharded = b.shard(mesh, P("tp"))
        
        result = add(a_sharded, b_sharded)
        expected = np_a + np_b
        
        assert_shape(result, (8, 16))
        assert_allclose(result, expected)


# =============================================================================
# Multi-axis Sharding
# =============================================================================

class TestBinaryOpsMultiAxisSharding:
    """Test binary ops with multi-axis (2D mesh) sharding."""
    
    def test_2d_mesh_both_axes_sharded(self, mesh_2d):
        """Both tensor axes sharded on 2D mesh."""
        np_a = make_array(8, 8, seed=42)
        np_b = make_array(8, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # Shard axis 0 on dp, axis 1 on tp
        a_sharded = a.shard(mesh_2d, P("dp", "tp"))
        b_sharded = b.shard(mesh_2d, P("dp", "tp"))
        
        result = add(a_sharded, b_sharded)
        expected = np_a + np_b
        
        assert_shape(result, (8, 8))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)
    
    def test_3d_tensor_on_2d_mesh(self, mesh_2d):
        """3D tensor with 2 axes sharded on 2D mesh."""
        np_a = make_array(4, 8, 16, seed=42)
        np_b = make_array(4, 8, 16, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        # Shard axis 0 on dp, axis 2 on tp
        a_sharded = a.shard(mesh_2d, P("dp", None, "tp"))
        b_sharded = b.shard(mesh_2d, P("dp", None, "tp"))
        
        result = mul(a_sharded, b_sharded)
        expected = np_a * np_b
        
        assert_shape(result, (4, 8, 16))
        assert_is_sharded(result, True)
        assert_allclose(result, expected)


# =============================================================================
# Sharding Propagation Verification
# =============================================================================

class TestShardingPropagationVerification:
    """Verify that output sharding matches expected based on input shardings."""
    
    def test_output_inherits_sharding_from_inputs(self, mesh_1d):
        """Verify output has correct sharding spec after binary op."""
        np_a = make_array(8, 4, seed=42)
        np_b = make_array(8, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh_1d, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh_1d, axis=0)
        
        result = add(a, b)
        
        # Verify result is sharded
        assert result._impl.is_sharded
        assert result._impl.sharding is not None
        
        # Verify sharding spec: axis 0 should be sharded on "dp"
        sharding = result._impl.sharding
        assert len(sharding.dim_specs) >= 2
        
        # The first dimension should have the dp axis
        assert "dp" in sharding.dim_specs[0].axes or sharding.dim_specs[0].axes == ["dp"]
    
    def test_output_sharding_matches_2d_mesh_input(self, mesh_2d):
        """Verify output sharding with 2D mesh input."""
        np_a = make_array(8, 8, seed=42)
        np_b = make_array(8, 8, seed=43)
        
        a = tensor_from_numpy(np_a)
        b = tensor_from_numpy(np_b)
        
        a_sharded = a.shard(mesh_2d, P("dp", "tp"))
        b_sharded = b.shard(mesh_2d, P("dp", "tp"))
        
        result = mul(a_sharded, b_sharded)
        
        # Verify result sharding
        assert result._impl.is_sharded
        sharding = result._impl.sharding
        
        # Both axes should be sharded
        assert sharding.dim_specs[0].axes  # First axis has sharding
        assert sharding.dim_specs[1].axes  # Second axis has sharding


__all__ = [
    "TestBinaryOpsSameSharding",
    "TestBinaryOpsShardedReplicated",
    "TestBinaryOpsMismatchedSharding",
    "TestBinaryOpsBroadcastSharding",
    "TestBinaryOpsMultiAxisSharding",
    "TestShardingPropagationVerification",
]
