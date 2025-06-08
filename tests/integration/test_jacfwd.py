#!/usr/bin/env python3
"""
Compare Nabla jacfwd with JAX jacfwd to ensure API compatibility
"""

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available - skipping comparison")

import numpy as np
import nabla as nb

def compare_simple_function():
    """Compare with JAX for simple single-input function."""
    print("\n=== Comparing Simple Function ===")
    
    def simple_func_jax(x):
        return x * x
    
    def simple_func_nabla(x):
        return x * x
    
    # Test input
    x_np = np.array([1.0, 2.0, 3.0])
    
    if JAX_AVAILABLE:
        # JAX computation
        x_jax = jnp.array(x_np)
        jac_jax = jax.jacfwd(simple_func_jax)(x_jax)
        print(f"JAX jacfwd result:\n{jac_jax}")
    
    # Nabla computation
    x_nabla = nb.array(x_np)
    jac_nabla_fn = nb.jacfwd(simple_func_nabla)
    jac_nabla = jac_nabla_fn(x_nabla)
    print(f"Nabla jacfwd result:\n{jac_nabla.to_numpy()}")
    
    if JAX_AVAILABLE:
        np.testing.assert_allclose(jac_nabla.to_numpy(), jac_jax, rtol=1e-6)
        print("✅ Simple function results match!")
    else:
        print("Expected: diagonal matrix [[2, 0, 0], [0, 4, 0], [0, 0, 6]]")

def compare_multi_input_function():
    """Compare with JAX for multi-input function."""
    print("\n=== Comparing Multi-input Function ===")
    
    def multi_func_jax(x, y):
        return x * y + x * x
    
    def multi_func_nabla(x, y):
        return x * y + x * x
    
    # Test inputs
    x_np = np.array([1.0, 2.0])
    y_np = np.array([3.0, 4.0])
    
    if JAX_AVAILABLE:
        # JAX computation for both arguments
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        jac_jax_both = jax.jacfwd(multi_func_jax, argnums=(0, 1))(x_jax, y_jax)
        print(f"JAX jacfwd result (both args):")
        for i, jac in enumerate(jac_jax_both):
            print(f"  arg {i}: {jac}")
        
        # JAX computation for x only
        jac_jax_x = jax.jacfwd(multi_func_jax, argnums=0)(x_jax, y_jax)
        print(f"JAX jacfwd result (x only):\n{jac_jax_x}")
    
    # Nabla computation for both arguments
    x_nabla = nb.array(x_np)
    y_nabla = nb.array(y_np)
    jac_nabla_fn_both = nb.jacfwd(multi_func_nabla, argnums=(0, 1))
    jac_nabla_both = jac_nabla_fn_both(x_nabla, y_nabla)
    print(f"Nabla jacfwd result (both args):")
    for i, jac in enumerate(jac_nabla_both):
        print(f"  arg {i}: {jac.to_numpy()}")
    
    # Nabla computation for x only
    jac_nabla_fn_x = nb.jacfwd(multi_func_nabla, argnums=0)
    jac_nabla_x = jac_nabla_fn_x(x_nabla, y_nabla)
    print(f"Nabla jacfwd result (x only):\n{jac_nabla_x.to_numpy()}")
    
    if JAX_AVAILABLE:
        # Compare both args case
        for i, (jax_jac, nabla_jac) in enumerate(zip(jac_jax_both, jac_nabla_both)):
            np.testing.assert_allclose(nabla_jac.to_numpy(), jax_jac, rtol=1e-6)
        print("✅ Multi-input (both args) results match!")
        
        # Compare x only case
        np.testing.assert_allclose(jac_nabla_x.to_numpy(), jac_jax_x, rtol=1e-6)
        print("✅ Multi-input (x only) results match!")
    else:
        print("Expected both args: ([[5, 0], [0, 8]], [[1, 0], [0, 2]])")
        print("Expected x only: [[5, 0], [0, 8]]")

def test_api_compatibility():
    """Test API compatibility with JAX."""
    print("\n=== Testing API Compatibility ===")
    
    def test_func(x):
        return nb.sum(x ** 2)
    
    x = nb.array([1.0, 2.0, 3.0])
    
    # Test different argnums formats
    try:
        # Test int argnum
        jac_fn_int = nb.jacfwd(test_func, argnums=0)
        result_int = jac_fn_int(x)
        print("✅ argnums=0 (int) works")
        
        # Test tuple argnum
        jac_fn_tuple = nb.jacfwd(test_func, argnums=(0,))
        result_tuple = jac_fn_tuple(x)
        print("✅ argnums=(0,) (tuple) works")
        
        # Test list argnum
        jac_fn_list = nb.jacfwd(test_func, argnums=[0])
        result_list = jac_fn_list(x)
        print("✅ argnums=[0] (list) works")
        
        # Check they all give same result
        np.testing.assert_allclose(result_int.to_numpy(), result_tuple.to_numpy())
        np.testing.assert_allclose(result_int.to_numpy(), result_list.to_numpy())
        print("✅ All argnum formats give identical results")
        
    except Exception as e:
        print(f"❌ API compatibility issue: {e}")

if __name__ == "__main__":
    print("=== JAX COMPATIBILITY TEST ===")
    
    if JAX_AVAILABLE:
        print("✓ JAX available for comparison")
    else:
        print("✗ JAX not available - will only test Nabla behavior")
    
    compare_simple_function()
    compare_multi_input_function()
    test_api_compatibility()
    
    print("\n=== COMPATIBILITY TEST COMPLETE ===")
