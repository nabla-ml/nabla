#!/usr/bin/env python3
"""
Test jacobian implementation against JAX as ground truth.

This validates that our adapted jacobian logic from the experimental Mojo code
produces the same results as JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


def test_vector_to_scalar_jacobian():
    """Test vector to scalar jacobian against JAX."""
    print("\nTesting vector to scalar jacobian vs JAX...")

    # Define function
    def func_nb(x):
        return x * x  # nb.sum(x * x)

    def func_jax(x):
        return x * x  # jnp.sum(x * x)

    # Test point
    x_nb = nb.array([1.0, 2.0, 3.0])
    x_jax = jnp.array([1.0, 2.0, 3.0])

    # Compute jacobians
    # print(nb.xpr(nb.jacrev(func_nb), x_nb))
    jac_nb = nb.jacrev(func_nb)(x_nb)
    jac_jax = jax.jacrev(func_jax)(x_jax)

    print(f"  Nabla jacobian: {jac_nb.to_numpy()}")
    print(f"  JAX jacobian: {np.array(jac_jax)}")
    print(f"  Shapes match: {jac_nb.shape == jac_jax.shape}")
    print(f"  Values match: {np.allclose(jac_nb.to_numpy(), jac_jax)}")

    assert jac_nb.shape == jac_jax.shape
    assert np.allclose(jac_nb.to_numpy(), jac_jax)
    print("✓ Vector to scalar jacobian test passed\n")

    # print(nb.xpr(nb.jacrev(nb.jacrev(func_nb)), x_nb))
    hess_nb = nb.jacrev(nb.jacrev(func_nb))(x_nb)
    hess_jax = jax.jacrev(jax.jacrev(func_jax))(x_jax)

    print(f"  Nabla hessian: {hess_nb.to_numpy()}")
    print(f"  JAX hessian: {np.array(hess_jax)}")
    print(f"  Hessian shapes match: {hess_nb.shape == hess_jax.shape}")
    print(f"  Hessian values match: {np.allclose(hess_nb.to_numpy(), hess_jax)}")
    assert hess_nb.shape == hess_jax.shape
    assert np.allclose(hess_nb.to_numpy(), hess_jax)

    # now to the third derivaitve
    print("Testing third derivative...")
    third_deriv_nb = nb.jacrev(nb.jacrev(nb.jacrev(func_nb)))(x_nb)
    third_deriv_jax = jax.jacrev(jax.jacrev(jax.jacrev(func_jax)))(x_jax)
    print(f"  Nabla third derivative: {third_deriv_nb.to_numpy()}")
    print(f"  JAX third derivative: {np.array(third_deriv_jax)}")
    print(
        f"  Third derivative shapes match: {third_deriv_nb.shape == third_deriv_jax.shape}"
    )
    print(
        f"  Third derivative values match: {np.allclose(third_deriv_nb.to_numpy(), third_deriv_jax)}"
    )
    assert third_deriv_nb.shape == third_deriv_jax.shape
    assert np.allclose(third_deriv_nb.to_numpy(), third_deriv_jax)


test_vector_to_scalar_jacobian()
