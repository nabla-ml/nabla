#!/usr/bin/env python3
"""Test the improved has_aux implementation."""

import nabla as nb
import numpy as np


def test_vjp_has_aux():
    """Test VJP with has_aux parameter."""
    print("=== Testing VJP with has_aux ===")
    
    def func_with_aux(x):
        """Function that returns (main_output, auxiliary_data)."""
        main_output = x ** 2
        aux_data = {"debug_info": "squared", "step": 42}
        return main_output, aux_data
    
    def func_without_aux(x):
        """Regular function."""
        return x ** 2
    
    x = nb.array([2.0, 3.0])
    
    # Test 1: Function without aux (has_aux=False)
    print("1. Testing function without aux (has_aux=False)")
    output1, vjp_fn1 = nb.vjp(func_without_aux, x, has_aux=False)
    cotangent = nb.array([1.0, 1.0])
    grads1 = vjp_fn1(cotangent)
    
    print(f"  Output: {output1}")
    print(f"  Gradients: {grads1}")
    print(f"  Expected gradients: [4.0, 6.0] (2*x)")
    
    # Test 2: Function with aux (has_aux=True)
    print("\n2. Testing function with aux (has_aux=True)")
    output2, vjp_fn2, aux = nb.vjp(func_with_aux, x, has_aux=True)
    grads2 = vjp_fn2(cotangent)
    
    print(f"  Output: {output2}")
    print(f"  Auxiliary data: {aux}")
    print(f"  Gradients: {grads2}")
    print(f"  Expected gradients: [4.0, 6.0] (2*x)")
    
    # Verify results are the same
    assert np.allclose(output1.to_numpy(), output2.to_numpy()), "Outputs should be identical"
    assert np.allclose(grads1[0].to_numpy(), grads2[0].to_numpy()), "Gradients should be identical"
    assert aux["debug_info"] == "squared", "Auxiliary data should be preserved"
    assert aux["step"] == 42, "Auxiliary data should be preserved"
    
    print("  ✓ VJP has_aux test passed!")


def test_jacrev_has_aux():
    """Test JACREV with has_aux parameter."""
    print("\n=== Testing JACREV with has_aux ===")
    
    def func_with_aux(x, y):
        """Function that returns (main_output, auxiliary_data)."""
        main_output = x * y + x ** 2
        aux_data = {"operation": "x*y + x^2", "inputs": [x.shape, y.shape]}
        return main_output, aux_data
    
    def func_without_aux(x, y):
        """Regular function."""
        return x * y + x ** 2
    
    x = nb.array([2.0])
    y = nb.array([3.0])
    
    # Test 1: Function without aux (has_aux=False)
    print("1. Testing JACREV without aux (has_aux=False)")
    jac_fn1 = nb.jacrev(func_without_aux, argnums=(0, 1), has_aux=False)
    jac1 = jac_fn1(x, y)
    
    print(f"  Jacobian: {jac1}")
    # Expected: df/dx = y + 2*x = 3 + 4 = 7, df/dy = x = 2
    
    # Test 2: Function with aux (has_aux=True)
    print("\n2. Testing JACREV with aux (has_aux=True)")
    jac_fn2 = nb.jacrev(func_with_aux, argnums=(0, 1), has_aux=True)
    jac2, aux = jac_fn2(x, y)
    
    print(f"  Jacobian: {jac2}")
    print(f"  Auxiliary data: {aux}")
    
    # Verify results are the same
    assert len(jac1) == len(jac2) == 2, "Should have gradients for both arguments"
    assert np.allclose(jac1[0].to_numpy(), jac2[0].to_numpy()), "Jacobians should be identical"
    assert np.allclose(jac1[1].to_numpy(), jac2[1].to_numpy()), "Jacobians should be identical"
    assert aux["operation"] == "x*y + x^2", "Auxiliary data should be preserved"
    
    print("  ✓ JACREV has_aux test passed!")


def test_error_handling():
    """Test error handling for has_aux."""
    print("\n=== Testing Error Handling ===")
    
    def bad_func(x):
        """Function that doesn't return a tuple when has_aux=True."""
        return x ** 2  # Should return (output, aux) when has_aux=True
    
    x = nb.array([2.0])
    
    try:
        # This should raise an error
        output, vjp_fn, aux = nb.vjp(bad_func, x, has_aux=True)
        assert False, "Should have raised an error"
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {e}")
        assert "must return a tuple (output, aux)" in str(e)
    
    try:
        # This should also raise an error for jacrev
        jac_fn = nb.jacrev(bad_func, has_aux=True)
        jac, aux = jac_fn(x)
        assert False, "Should have raised an error"
    except ValueError as e:
        print(f"  ✓ Correctly caught JACREV error: {e}")
        assert "must return a tuple (output, aux)" in str(e)


if __name__ == "__main__":
    test_vjp_has_aux()
    test_jacrev_has_aux()
    test_error_handling()
    print("\n🎉 All has_aux tests passed!")
