import jax
import jax.numpy as jnp
import numpy as np
import endia as nb


def test_vector_to_scalar_jacobian():
    """Test vector to scalar jacobian against JAX."""

    # Define functions for both Endia and JAX
    def func_nb(x):
        # return (x * x).sum([0]) #y#x * x * y * y * y
        return x * x * x * x * x#nd.sin(x)#nd.unsqueeze(x, [0])

    def func_jax(x):
        # return (x * x).sum([0])#y#x * x * y * y * y
        return x * x * x * x * x#jnp.sin(x)#jnp.expand_dims(x, axis=0)

    # Test data - use compatible shapes and values
    x_nb = nd.arange((2, 3))
    y_nb = nd.arange((2, 3))
    
    # Convert to JAX arrays with same values
    x_jax = jnp.array(x_nd.to_numpy())
    y_jax = jnp.array(y_nd.to_numpy())

    print("=== INPUT COMPARISON ===")
    print(f"x_nb shape: {x_nd.shape}, x_jax shape: {x_jax.shape}")
    print(f"y_nb shape: {y_nd.shape}, y_jax shape: {y_jax.shape}")
    # print(f"Input values match: {np.allclose(x_nd.to_numpy(), x_jax)}")

    # Forward pass comparison
    print("\n=== FORWARD PASS COMPARISON ===")
    result_nb = func_nb(x_nb)
    result_jax = func_jax(x_jax)
    print("Endia XPR:")
    print(nd.xpr(func_nb, x_nb))
    print(f"Endia result shape: {result_nd.shape}")
    print(f"JAX result shape: {result_jax.shape}")
    print(f"Forward values match: {np.allclose(result_nd.to_numpy(), result_jax)}")

    if not np.allclose(result_nd.to_numpy(), result_jax):
        print("WARNING: Forward pass values don't match!")
        max_diff = np.max(np.abs(result_nd.to_numpy() - result_jax))
        print(f"Max difference: {max_diff}")
        # Show sample values for debugging
        print(f"Endia sample values: {result_nd.to_numpy().flat[:10]}")
        print(f"JAX sample values: {result_jax.flatten()[:10]}")
        print(f"Difference sample: {(result_nd.to_numpy() - result_jax).flatten()[:10]}")


    # First derivative (Jacobian) comparison
    print("\n=== FIRST DERIVATIVE (JACOBIAN) COMPARISON ===")
    jac_fn_nb = nd.jacrev(func_nb)
    jac_fn_jax = jax.jacrev(func_jax)
    
    jac_nb = jac_fn_nb(x_nb)
    jac_jax = jac_fn_jax(x_jax)
    
    print("Endia XPR:")
    print(nd.xpr(jac_fn_nb, x_nb))
    print(f"Endia jacobian shape: {jac_nd.shape}")
    print(f"JAX jacobian shape: {jac_jax.shape}")
    print(f"Jacobian values match: {np.allclose(jac_nd.to_numpy(), jac_jax, rtol=1e-5, atol=1e-6)}")
    
    if not np.allclose(jac_nd.to_numpy(), jac_jax, rtol=1e-5, atol=1e-6):
        print("WARNING: Jacobian values don't match!")
        max_diff = np.max(np.abs(jac_nd.to_numpy() - jac_jax))
        print(f"Max difference: {max_diff}")


    # Second derivative (Hessian) comparison
    print("\n=== SECOND DERIVATIVE (HESSIAN) COMPARISON ===")
    hess_fn_nb = nd.jacrev(jac_fn_nb)
    hess_fn_jax = jax.jacrev(jac_fn_jax)
    
    hess_nb = hess_fn_nb(x_nb)
    hess_jax = hess_fn_jax(x_jax)

    print("Endia XPR:")
    print(nd.xpr(hess_fn_nb, x_nb))
    print(f"Endia hessian shape: {hess_nd.shape}")
    print(f"JAX hessian shape: {hess_jax.shape}")
    print(f"Hessian values match: {np.allclose(hess_nd.to_numpy(), hess_jax, rtol=1e-4, atol=1e-5)}")
    
    if not np.allclose(hess_nd.to_numpy(), hess_jax, rtol=1e-4, atol=1e-5):
        print("WARNING: Hessian values don't match!")
        max_diff = np.max(np.abs(hess_nd.to_numpy() - hess_jax))
        print(f"Max difference: {max_diff}")


    # Third derivative comparison
    print("\n=== THIRD DERIVATIVE COMPARISON ===")
    third_fn_nb = nd.jacrev(hess_fn_nb)
    third_fn_jax = jax.jacrev(hess_fn_jax)
    
    third_nb = third_fn_nb(x_nb)
    third_jax = third_fn_jax(x_jax)
    
    print("Endia XPR:")
    print(nd.xpr(third_fn_nb, x_nb))
    print(f"Endia third derivative shape: {third_nd.shape}")
    print(f"JAX third derivative shape: {third_jax.shape}")
    print(f"Third derivative values match: {np.allclose(third_nd.to_numpy(), third_jax, rtol=1e-3, atol=1e-4)}")
    
    if not np.allclose(third_nd.to_numpy(), third_jax, rtol=1e-3, atol=1e-4):
        print("WARNING: Third derivative values don't match!")
        max_diff = np.max(np.abs(third_nd.to_numpy() - third_jax))
        print(f"Max difference: {max_diff}")
        # Show sample values for debugging
        print(f"Endia sample values: {third_nd.to_numpy().flat[:10]}")
        print(f"JAX sample values: {third_jax.flatten()[:10]}")
        print(f"Difference sample: {(third_nd.to_numpy() - third_jax).flatten()[:10]}")


    # Forth derivative comparison
    print("\n=== FORTH DERIVATIVE COMPARISON ===")
    forth_fn_nb = nd.jacrev(third_fn_nb)
    forth_fn_jax = jax.jacrev(third_fn_jax)
    forth_nb = forth_fn_nb(x_nb)
    forth_jax = forth_fn_jax(x_jax)

    print("Endia XPR:")
    print(nd.xpr(forth_fn_nb, x_nb))
    print("\nJAXPR:")
    print(jax.make_jaxpr(forth_fn_jax)(x_jax))
    print(f"Endia forth derivative shape: {forth_nd.shape}")
    print(f"JAX forth derivative shape: {forth_jax.shape}")
    print(f"Forth derivative values match: {np.allclose(forth_nd.to_numpy(), forth_jax, rtol=1e-3, atol=1e-4)}")

    if not np.allclose(forth_nd.to_numpy(), forth_jax, rtol=1e-3, atol=1e-4):
        print("WARNING: Forth derivative values don't match!")
        max_diff = np.max(np.abs(forth_nd.to_numpy() - forth_jax))
        print(f"Max difference: {max_diff}")
        # Show sample values for debugging
        print(f"Endia sample values: {forth_nd.to_numpy().flat[:10]}")
        print(f"JAX sample values: {forth_jax.flatten()[:10]}")
        print(f"Difference sample: {(forth_nd.to_numpy() - forth_jax).flatten()[:10]}")
    
    # Final assertions for test validation
    assert np.allclose(result_nd.to_numpy(), result_jax), "Forward pass must match"
    assert np.allclose(jac_nd.to_numpy(), jac_jax, rtol=1e-5, atol=1e-6), "Jacobian must match"
    assert np.allclose(hess_nd.to_numpy(), hess_jax, rtol=1e-4, atol=1e-5), "Hessian must match"
    assert np.allclose(third_nd.to_numpy(), third_jax, rtol=1e-3, atol=1e-4), "Third derivative must match"
    assert np.allclose(forth_nd.to_numpy(), forth_jax, rtol=1e-3, atol=1e-4), "Forth derivative must match"
    
    print("\n✅ All tests passed! Endia's derivatives match JAX ground truth.")


if __name__ == "__main__":
    test_vector_to_scalar_jacobian()