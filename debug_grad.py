#!/usr/bin/env python3

import nabla as nb
import numpy as np
import jax
import jax.numpy as jnp

def simple_test():
    """Debug the grad function issue."""
    
    # Define simple function  
    def func_nb(x):
        return (nb.sin(x) * x).sum()
    
    def func_jax(x):
        return (jnp.sin(x) * x).sum()
    
    # Test data
    x_nb = nb.arange((2, 3))
    x_jax = jnp.array(x_nb.to_numpy())
    
    print(f"Input shape: {x_nb.shape}")
    
    # Check function output
    result_nb = func_nb(x_nb)
    result_jax = func_jax(x_jax)
    
    print(f"Function output shape - Nabla: {result_nb.shape}, JAX: {result_jax.shape}")
    print(f"Function values match: {np.allclose(result_nb.to_numpy(), result_jax)}")
    
    # Check gradients
    grad_fn_nb = nb.grad(func_nb)
    grad_fn_jax = jax.grad(func_jax)
    
    print("\n=== DEBUGGING GRADIENT ===")
    
    # Let's trace the gradient computation to see what's happening
    print("Nabla gradient XPR:")
    print(nb.xpr(grad_fn_nb, x_nb))
    
    grad_nb = grad_fn_nb(x_nb)
    grad_jax = grad_fn_jax(x_jax)
    
    print(f"Gradient shape - Nabla: {grad_nb.shape}, JAX: {grad_jax.shape}")
    print(f"Gradient values - Nabla: {grad_nb.to_numpy()}")
    print(f"Gradient values - JAX: {grad_jax}")
    
if __name__ == "__main__":
    simple_test()
