# ===----------------------------------------------------------------------=== #
# Nabla 2025
# ===----------------------------------------------------------------------=== #
"""Comprehensive vmap + sharding tests across 1D, 2D, and 3D meshes.

This file systematically tests all user-facing operations with:
1. vmap on 1D meshes (2, 4, 8 devices)
2. vmap on 2D meshes ((2,2), (2,4), (4,2), (1,4))
3. vmap on 3D meshes ((2,2,2), (2,2,4), (4,2,2))
4. Different sharding patterns (batch on various mesh axes)
5. Multi-axis sharding scenarios

Test pattern:
- Create batched input with shape (batch, *op_shape)
- Shard on batch dimension using different mesh configurations
- Apply vmap(fn) where fn operates on op_shape
- Verify numerical correctness against numpy reference
"""

import pytest
import numpy as np

import nabla
from nabla import vmap, DeviceMesh
from nabla import (
    add, sub, mul, div,
    relu, sigmoid, tanh, exp, neg,
    reduce_sum, mean,
    reshape, squeeze, unsqueeze, swap_axes,
    matmul,
)
from nabla.sharding.spec import DimSpec

from .conftest import (
    make_array, make_positive_array, tensor_from_numpy, to_numpy,
    assert_allclose, assert_shape, assert_is_sharded,
    shard_on_axis, shard_on_axes, replicated,
)


# =============================================================================
# 1D Mesh Tests - Various Sizes
# =============================================================================

class TestVmapSharding1DMeshes:
    """Test vmap+sharding with 1D meshes of different sizes."""
    
    @pytest.mark.parametrize("mesh_size", [2, 4, 8])
    def test_vmap_relu_1d_mesh(self, mesh_size):
        """vmap(relu) on 1D mesh with varying sizes."""
        batch = mesh_size * 2  # Ensure divisible by mesh size
        mesh = DeviceMesh(f"mesh_1d_{mesh_size}", (mesh_size,), ("dp",))
        
        np_x = make_array(batch, 4, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0)
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_size", [2, 4, 8])
    def test_vmap_add_1d_mesh(self, mesh_size):
        """vmap(add) on 1D mesh with varying sizes."""
        batch = mesh_size * 2
        mesh = DeviceMesh(f"mesh_1d_{mesh_size}", (mesh_size,), ("dp",))
        
        np_a = make_array(batch, 4, 8, seed=42)
        np_b = make_array(batch, 4, 8, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0)
        
        result = vmap(add)(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_size", [2, 4, 8])
    def test_vmap_matmul_1d_mesh(self, mesh_size):
        """vmap(matmul) on 1D mesh with varying sizes."""
        batch = mesh_size * 2
        mesh = DeviceMesh(f"mesh_1d_{mesh_size}", (mesh_size,), ("dp",))
        
        np_a = make_array(batch, 8, 16, seed=42)
        np_b = make_array(batch, 16, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0)
        
        result = vmap(matmul)(a, b)
        expected = np.matmul(np_a, np_b)
        
        assert_shape(result, (batch, 8, 4))
        assert_allclose(result, expected, rtol=1e-4)
    
    @pytest.mark.parametrize("mesh_size", [2, 4, 8])
    def test_vmap_reduce_sum_1d_mesh(self, mesh_size):
        """vmap(reduce_sum) on 1D mesh with varying sizes."""
        batch = mesh_size * 2
        mesh = DeviceMesh(f"mesh_1d_{mesh_size}", (mesh_size,), ("dp",))
        
        np_x = make_array(batch, 8, 16, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0)
        
        def fn(t):
            return reduce_sum(t, axis=1)  # Reduce last logical axis
        
        result = vmap(fn)(x)
        expected = np.sum(np_x, axis=2)  # Physical axis 2
        
        assert_shape(result, (batch, 8))
        assert_allclose(result, expected)


# =============================================================================
# 2D Mesh Tests - Various Shapes
# =============================================================================

class TestVmapSharding2DMeshes:
    """Test vmap+sharding with 2D meshes of different shapes."""
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2), (2, 4), (4, 2), (1, 4)])
    def test_vmap_relu_2d_mesh_batch_on_dp(self, mesh_shape):
        """vmap(relu) with batch sharded on 'dp' axis of 2D mesh."""
        dp_size, tp_size = mesh_shape
        batch = dp_size * 2  # Divisible by dp
        mesh = DeviceMesh("mesh_2d", mesh_shape, ("dp", "tp"))
        
        np_x = make_array(batch, 4, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)  # dp
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2), (2, 4), (4, 2)])
    def test_vmap_relu_2d_mesh_batch_on_tp(self, mesh_shape):
        """vmap(relu) with batch sharded on 'tp' axis of 2D mesh."""
        dp_size, tp_size = mesh_shape
        batch = tp_size * 2  # Divisible by tp
        mesh = DeviceMesh("mesh_2d", mesh_shape, ("dp", "tp"))
        
        np_x = make_array(batch, 4, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=1)  # tp
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2), (2, 4), (4, 2)])
    def test_vmap_add_2d_mesh(self, mesh_shape):
        """vmap(add) on 2D mesh."""
        dp_size, tp_size = mesh_shape
        batch = dp_size * 2
        mesh = DeviceMesh("mesh_2d", mesh_shape, ("dp", "tp"))
        
        np_a = make_array(batch, 4, 8, seed=42)
        np_b = make_array(batch, 4, 8, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0, mesh_axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0, mesh_axis=0)
        
        result = vmap(add)(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2), (2, 4), (4, 2)])
    def test_vmap_matmul_2d_mesh(self, mesh_shape):
        """vmap(matmul) on 2D mesh."""
        dp_size, tp_size = mesh_shape
        batch = dp_size * 2
        mesh = DeviceMesh("mesh_2d", mesh_shape, ("dp", "tp"))
        
        np_a = make_array(batch, 8, 16, seed=42)
        np_b = make_array(batch, 16, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0, mesh_axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0, mesh_axis=0)
        
        result = vmap(matmul)(a, b)
        expected = np.matmul(np_a, np_b)
        
        assert_shape(result, (batch, 8, 4))
        assert_allclose(result, expected, rtol=1e-4)
    
    def test_vmap_with_multi_axis_sharding_2d(self, mesh_2d):
        """vmap where batch is on dp, and feature dim on tp."""
        # Shape: (batch=4, seq=8, hidden=4)
        # Shard: batch on dp (size 2), hidden on tp (size 2)
        np_x = make_array(4, 8, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axes(x, mesh_2d, {0: 0, 2: 1})  # batch->dp, hidden->tp
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (4, 8, 4))
        assert_allclose(result, expected)


# =============================================================================
# 3D Mesh Tests - Various Shapes
# =============================================================================

class TestVmapSharding3DMeshes:
    """Test vmap+sharding with 3D meshes of different shapes."""
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2, 2), (2, 2, 4), (4, 2, 2)])
    def test_vmap_relu_3d_mesh_batch_on_dp(self, mesh_shape):
        """vmap(relu) with batch sharded on 'dp' axis of 3D mesh."""
        dp_size, tp_size, pp_size = mesh_shape
        batch = dp_size * 2
        mesh = DeviceMesh("mesh_3d", mesh_shape, ("dp", "tp", "pp"))
        
        np_x = make_array(batch, 4, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)  # dp
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2, 2), (2, 2, 4), (4, 2, 2)])
    def test_vmap_relu_3d_mesh_batch_on_tp(self, mesh_shape):
        """vmap(relu) with batch sharded on 'tp' axis of 3D mesh."""
        dp_size, tp_size, pp_size = mesh_shape
        batch = tp_size * 2
        mesh = DeviceMesh("mesh_3d", mesh_shape, ("dp", "tp", "pp"))
        
        np_x = make_array(batch, 4, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=1)  # tp
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2, 2), (2, 2, 4), (4, 2, 2)])
    def test_vmap_relu_3d_mesh_batch_on_pp(self, mesh_shape):
        """vmap(relu) with batch sharded on 'pp' axis of 3D mesh."""
        dp_size, tp_size, pp_size = mesh_shape
        batch = pp_size * 2
        mesh = DeviceMesh("mesh_3d", mesh_shape, ("dp", "tp", "pp"))
        
        np_x = make_array(batch, 4, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=2)  # pp
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2, 2), (2, 2, 4)])
    def test_vmap_add_3d_mesh(self, mesh_shape):
        """vmap(add) on 3D mesh."""
        dp_size, tp_size, pp_size = mesh_shape
        batch = dp_size * 2
        mesh = DeviceMesh("mesh_3d", mesh_shape, ("dp", "tp", "pp"))
        
        np_a = make_array(batch, 4, 8, seed=42)
        np_b = make_array(batch, 4, 8, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0, mesh_axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0, mesh_axis=0)
        
        result = vmap(add)(a, b)
        expected = np_a + np_b
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape", [(2, 2, 2), (2, 2, 4)])
    def test_vmap_matmul_3d_mesh(self, mesh_shape):
        """vmap(matmul) on 3D mesh."""
        dp_size, tp_size, pp_size = mesh_shape
        batch = dp_size * 2
        mesh = DeviceMesh("mesh_3d", mesh_shape, ("dp", "tp", "pp"))
        
        np_a = make_array(batch, 8, 16, seed=42)
        np_b = make_array(batch, 16, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0, mesh_axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0, mesh_axis=0)
        
        result = vmap(matmul)(a, b)
        expected = np.matmul(np_a, np_b)
        
        assert_shape(result, (batch, 8, 4))
        assert_allclose(result, expected, rtol=1e-4)
    
    def test_vmap_with_multi_axis_sharding_3d(self, mesh_3d):
        """vmap where different tensor dims are on different mesh axes."""
        # Shape: (batch=4, seq=4, hidden=4)
        # Shard: batch on dp (2), seq on tp (2), hidden on pp (2)
        np_x = make_array(4, 4, 4, seed=42)
        x = tensor_from_numpy(np_x)
        x = shard_on_axes(x, mesh_3d, {0: 0, 1: 1, 2: 2})  # batch->dp, seq->tp, hidden->pp
        
        result = vmap(relu)(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (4, 4, 4))
        assert_allclose(result, expected)


# =============================================================================
# All User-Facing Unary Ops - Parametrized Mesh Testing
# =============================================================================

class TestVmapAllUnaryOpsAllMeshes:
    """Test all unary ops with vmap across mesh configurations."""
    
    UNARY_OPS = [
        ("relu", relu, lambda x: np.maximum(x, 0)),
        ("sigmoid", sigmoid, lambda x: 1 / (1 + np.exp(-x))),
        ("tanh", tanh, np.tanh),
        ("exp", exp, np.exp),
        ("neg", neg, lambda x: -x),
    ]
    
    MESH_CONFIGS = [
        # 1D meshes
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        ((8,), ("dp",)),
        # 2D meshes
        ((2, 2), ("dp", "tp")),
        ((2, 4), ("dp", "tp")),
        ((4, 2), ("dp", "tp")),
        # 3D meshes
        ((2, 2, 2), ("dp", "tp", "pp")),
        ((2, 2, 4), ("dp", "tp", "pp")),
    ]
    
    @pytest.mark.parametrize("op_name,nabla_op,numpy_op", UNARY_OPS)
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_unary_op(self, op_name, nabla_op, numpy_op, mesh_shape, mesh_axes):
        """Test each unary op with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2  # Use first mesh dim for batch sharding
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        # Use smaller values for exp to avoid overflow
        if op_name == "exp":
            np_x = make_array(batch, 4, 8, seed=42) * 0.1
        else:
            np_x = make_array(batch, 4, 8, seed=42)
        
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        result = vmap(nabla_op)(x)
        expected = numpy_op(np_x)
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected, rtol=1e-4)


# =============================================================================
# All User-Facing Binary Ops - Parametrized Mesh Testing
# =============================================================================

class TestVmapAllBinaryOpsAllMeshes:
    """Test all binary ops with vmap across mesh configurations."""
    
    BINARY_OPS = [
        ("add", add, lambda a, b: a + b),
        ("sub", sub, lambda a, b: a - b),
        ("mul", mul, lambda a, b: a * b),
        ("div", div, lambda a, b: a / b),
    ]
    
    MESH_CONFIGS = [
        # 1D meshes
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        # 2D meshes
        ((2, 2), ("dp", "tp")),
        ((2, 4), ("dp", "tp")),
        # 3D meshes
        ((2, 2, 2), ("dp", "tp", "pp")),
    ]
    
    @pytest.mark.parametrize("op_name,nabla_op,numpy_op", BINARY_OPS)
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_binary_op(self, op_name, nabla_op, numpy_op, mesh_shape, mesh_axes):
        """Test each binary op with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_a = make_array(batch, 4, 8, seed=42)
        # For div, use positive values to avoid division by zero
        if op_name == "div":
            np_b = make_positive_array(batch, 4, 8, seed=43)
        else:
            np_b = make_array(batch, 4, 8, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0, mesh_axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0, mesh_axis=0)
        
        result = vmap(nabla_op)(a, b)
        expected = numpy_op(np_a, np_b)
        
        assert_shape(result, (batch, 4, 8))
        assert_allclose(result, expected)


# =============================================================================
# Reduction Ops - Parametrized Mesh Testing
# =============================================================================

class TestVmapReductionOpsAllMeshes:
    """Test reduction ops with vmap across mesh configurations."""
    
    MESH_CONFIGS = [
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        ((2, 2), ("dp", "tp")),
        ((2, 2, 2), ("dp", "tp", "pp")),
    ]
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_reduce_sum(self, mesh_shape, mesh_axes):
        """Test reduce_sum with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, 8, 16, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        def fn(t):
            return reduce_sum(t, axis=1)
        
        result = vmap(fn)(x)
        expected = np.sum(np_x, axis=2)
        
        assert_shape(result, (batch, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_mean(self, mesh_shape, mesh_axes):
        """Test mean with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, 8, 16, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        def fn(t):
            return mean(t, axis=1)
        
        result = vmap(fn)(x)
        expected = np.mean(np_x, axis=2)
        
        assert_shape(result, (batch, 8))
        assert_allclose(result, expected)


# =============================================================================
# Matmul - Parametrized Mesh Testing
# =============================================================================

class TestVmapMatmulAllMeshes:
    """Test matmul with vmap across mesh configurations."""
    
    MESH_CONFIGS = [
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        ((2, 2), ("dp", "tp")),
        ((2, 4), ("dp", "tp")),
        ((2, 2, 2), ("dp", "tp", "pp")),
    ]
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_matmul(self, mesh_shape, mesh_axes):
        """Test matmul with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_a = make_array(batch, 8, 16, seed=42)
        np_b = make_array(batch, 16, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0, mesh_axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0, mesh_axis=0)
        
        result = vmap(matmul)(a, b)
        expected = np.matmul(np_a, np_b)
        
        assert_shape(result, (batch, 8, 4))
        assert_allclose(result, expected, rtol=1e-4)


# =============================================================================
# View Ops - Parametrized Mesh Testing
# =============================================================================

class TestVmapViewOpsAllMeshes:
    """Test view ops with vmap across mesh configurations."""
    
    MESH_CONFIGS = [
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        ((2, 2), ("dp", "tp")),
        ((2, 2, 2), ("dp", "tp", "pp")),
    ]
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_reshape(self, mesh_shape, mesh_axes):
        """Test reshape with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, 8, 16, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        def fn(t):
            return reshape(t, (128,))
        
        result = vmap(fn)(x)
        expected = np_x.reshape(batch, 128)
        
        assert_shape(result, (batch, 128))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_squeeze(self, mesh_shape, mesh_axes):
        """Test squeeze with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, 1, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        def fn(t):
            return squeeze(t, axis=0)
        
        result = vmap(fn)(x)
        expected = np_x.reshape(batch, 8)
        
        assert_shape(result, (batch, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_unsqueeze(self, mesh_shape, mesh_axes):
        """Test unsqueeze with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        def fn(t):
            return unsqueeze(t, axis=0)
        
        result = vmap(fn)(x)
        expected = np_x.reshape(batch, 1, 8)
        
        assert_shape(result, (batch, 1, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", MESH_CONFIGS)
    def test_vmap_swap_axes(self, mesh_shape, mesh_axes):
        """Test swap_axes with vmap on each mesh configuration."""
        batch = mesh_shape[0] * 2
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, 8, 16, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        def fn(t):
            return swap_axes(t, 0, 1)
        
        result = vmap(fn)(x)
        expected = np.swapaxes(np_x, 1, 2)
        
        assert_shape(result, (batch, 16, 8))
        assert_allclose(result, expected)


# =============================================================================
# Nested vmap + Sharding
# =============================================================================

class TestNestedVmapSharding:
    """Test nested vmap with various mesh configurations."""
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", [
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        ((2, 2), ("dp", "tp")),
    ])
    def test_nested_vmap_relu_sharded(self, mesh_shape, mesh_axes):
        """Nested vmap(vmap(relu)) on sharded input."""
        batch1 = mesh_shape[0] * 2
        batch2 = 4
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch1, batch2, 8, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        result = vmap(vmap(relu))(x)
        expected = np.maximum(np_x, 0)
        
        assert_shape(result, (batch1, batch2, 8))
        assert_allclose(result, expected)
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", [
        ((2,), ("dp",)),
        ((2, 2), ("dp", "tp")),
    ])
    def test_nested_vmap_matmul_sharded(self, mesh_shape, mesh_axes):
        """Nested vmap(vmap(matmul)) on sharded inputs."""
        batch1 = mesh_shape[0] * 2
        batch2 = 4
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_a = make_array(batch1, batch2, 8, 16, seed=42)
        np_b = make_array(batch1, batch2, 16, 4, seed=43)
        
        a = shard_on_axis(tensor_from_numpy(np_a), mesh, axis=0, mesh_axis=0)
        b = shard_on_axis(tensor_from_numpy(np_b), mesh, axis=0, mesh_axis=0)
        
        result = vmap(vmap(matmul))(a, b)
        expected = np.matmul(np_a, np_b)
        
        assert_shape(result, (batch1, batch2, 8, 4))
        assert_allclose(result, expected, rtol=1e-4)


# =============================================================================
# Composite Functions - Realistic Scenarios
# =============================================================================

class TestVmapCompositeSharded:
    """Test vmap on composite functions with sharding."""
    
    @pytest.mark.xfail(reason="Known limitation: unsharded tensors (w, b) captured in vmap with sharded input need proper broadcast handling")
    @pytest.mark.parametrize("mesh_shape,mesh_axes", [
        ((2,), ("dp",)),
        ((4,), ("dp",)),
        ((2, 2), ("dp", "tp")),
        ((2, 2, 2), ("dp", "tp", "pp")),
    ])
    def test_mlp_layer_sharded(self, mesh_shape, mesh_axes):
        """vmap over MLP layer: relu(x @ W + b) with sharded batch."""
        batch = mesh_shape[0] * 2
        in_features = 16
        out_features = 8
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, in_features, seed=42)
        np_w = make_array(in_features, out_features, seed=43)
        np_b = make_array(out_features, seed=44)
        
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        w = tensor_from_numpy(np_w)
        b = tensor_from_numpy(np_b)
        
        def mlp_layer(inputs):
            return relu(add(matmul(inputs, w), b))
        
        result = vmap(mlp_layer)(x)
        expected = np.maximum(np_x @ np_w + np_b, 0)
        
        assert_shape(result, (batch, out_features))
        assert_allclose(result, expected, rtol=1e-4)
    
    @pytest.mark.parametrize("mesh_shape,mesh_axes", [
        ((2,), ("dp",)),
        ((2, 2), ("dp", "tp")),
    ])
    def test_normalize_sharded(self, mesh_shape, mesh_axes):
        """vmap over normalization: x - mean(x) with sharded batch."""
        batch = mesh_shape[0] * 2
        features = 8
        mesh = DeviceMesh(f"mesh_{len(mesh_shape)}d", mesh_shape, mesh_axes)
        
        np_x = make_array(batch, features, seed=42)
        x = shard_on_axis(tensor_from_numpy(np_x), mesh, axis=0, mesh_axis=0)
        
        def normalize(inputs):
            m = mean(inputs, axis=0)
            return sub(inputs, m)
        
        result = vmap(normalize)(x)
        expected = np_x - np.mean(np_x, axis=1, keepdims=True)
        
        assert_shape(result, (batch, features))
        assert_allclose(result, expected)
