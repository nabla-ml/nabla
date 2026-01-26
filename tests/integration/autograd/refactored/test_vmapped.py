
import numpy as np
import pytest
import nabla as nb
from nabla.transforms import vmap
from nabla.core.sharding import DeviceMesh, DimSpec
from .utils import check_vjp, HAS_JAX

np.random.seed(42)

def test_vmap_basic_unary():
    print("\n=== Testing VMap Basic Unary ===")
    
    x = np.random.randn(8, 4).astype(np.float32)
    
    check_vjp(
        "vmap_relu",
        nb.ops.relu,
        pytest.importorskip("jax.nn").relu if HAS_JAX else None,
        (x,),
        vmap_axes=(0, 0)
    )

def test_vmap_basic_binary():
    print("\n=== Testing VMap Basic Binary ===")
    
    x1 = np.random.randn(8, 4).astype(np.float32)
    x2 = np.random.randn(8, 4).astype(np.float32)
    
    check_vjp(
        "vmap_add",
        nb.add,
        np.add if HAS_JAX else None,
        (x1, x2),
        vmap_axes=(0, 0)
    )

def test_vmap_matmul():
    print("\n=== Testing VMap Matmul ===")
    
    B, M, K, N = 8, 4, 3, 5
    x = np.random.randn(B, M, K).astype(np.float32)
    y = np.random.randn(B, K, N).astype(np.float32)
    
    check_vjp(
        "vmap_matmul",
        nb.matmul,
        np.matmul if HAS_JAX else None,
        (x, y),
        vmap_axes=(0, 0)
    )

def test_nested_vmap():
    print("\n=== Testing Nested VMap ===")
    
    shape = (2, 4, 3)
    x = np.random.randn(*shape).astype(np.float32)
    
    def inner_fn(t):
        return nb.ops.exp(t)
    
    def outer_fn(t):
        return vmap(inner_fn, in_axes=0, out_axes=0)(t)
    
    check_vjp(
        "nested_vmap_exp",
        outer_fn, 
        np.exp if HAS_JAX else None,
        (x,),
        vmap_axes=(0, 0)
    )

def test_vmap_sharded_spmd():
    print("\n=== Testing VMap + Sharded (SPMD) ===")
    
    mesh = DeviceMesh("mesh_vmap", (2,), ("dp",))
    
    shape = (8, 4)
    x = np.random.randn(*shape).astype(np.float32)
    
    spec = [DimSpec(["dp"]), DimSpec([])]
    
    check_vjp(
        "vmap_sharded_relu",
        nb.ops.relu,
        pytest.importorskip("jax.nn").relu if HAS_JAX else None,
        (x,),
        mesh=mesh,
        input_specs=(spec,),
        vmap_axes=(0, 0)
    )

if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
