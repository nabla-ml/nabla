#!/usr/bin/env python3
"""
Advanced jacfwd tests with complex scenarios against JAX ground truth
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


def test_multiple_inputs_multiple_outputs():
    """Test function with multiple inputs and multiple outputs."""
    print("\n=== Multiple Inputs, Multiple Outputs ===")

    def multi_io_func_jax(x, y):
        # Returns two outputs: [x*y + x^2, x + y^2]
        return jnp.array([x[0] * y[0] + x[0] ** 2, x[1] + y[1] ** 2])

    def multi_io_func_nabla(x, y):
        # Returns two outputs: [x*y + x^2, x + y^2]
        output1 = x[0] * y[0] + x[0] ** 2
        output2 = x[1] + y[1] ** 2
        return nb.concatenate(
            [nb.reshape(output1, (1,)), nb.reshape(output2, (1,))], axis=0
        )

    # Test inputs
    x_np = np.array([2.0, 3.0])
    y_np = np.array([4.0, 5.0])
    x_nabla = nb.array(x_np)
    y_nabla = nb.array(y_np)

    if JAX_AVAILABLE:
        # JAX computation
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)

        # Jacobian w.r.t. both arguments
        jac_jax_both = jax.jacfwd(multi_io_func_jax, argnums=(0, 1))(x_jax, y_jax)
        print("JAX jacfwd result (both args):")
        for i, jac in enumerate(jac_jax_both):
            print(f"  d/d_arg{i}: shape {jac.shape}\n{jac}")

        # Jacobian w.r.t. x only
        jac_jax_x = jax.jacfwd(multi_io_func_jax, argnums=0)(x_jax, y_jax)
        print(f"JAX jacfwd result (x only): shape {jac_jax_x.shape}\n{jac_jax_x}")

    try:
        # Jacobian w.r.t. both arguments
        jac_nabla_fn_both = nb.jacfwd(multi_io_func_nabla, argnums=(0, 1))
        jac_nabla_both = jac_nabla_fn_both(x_nabla, y_nabla)
        print("Nabla jacfwd result (both args):")
        for i, jac in enumerate(jac_nabla_both):
            print(f"  d/d_arg{i}: shape {jac.shape}\n{jac.to_numpy()}")

        # Jacobian w.r.t. x only
        jac_nabla_fn_x = nb.jacfwd(multi_io_func_nabla, argnums=0)
        jac_nabla_x = jac_nabla_fn_x(x_nabla, y_nabla)
        print(
            f"Nabla jacfwd result (x only): shape {jac_nabla_x.shape}\n{jac_nabla_x.to_numpy()}"
        )

        if JAX_AVAILABLE:
            # Compare results
            for _i, (jax_jac, nabla_jac) in enumerate(
                zip(jac_jax_both, jac_nabla_both, strict=False)
            ):
                np.testing.assert_allclose(nabla_jac.to_numpy(), jax_jac, rtol=1e-6)
            print("✅ Multi-input/output (both args) results match!")

            np.testing.assert_allclose(jac_nabla_x.to_numpy(), jac_jax_x, rtol=1e-6)
            print("✅ Multi-input/output (x only) results match!")

    except Exception as e:
        print(f"❌ Multi-input/output failed: {e}")
        import traceback

        traceback.print_exc()


def test_matrix_function():
    """Test function that operates on matrices using available Nabla operations."""
    print("\n=== Matrix Function ===")

    def matrix_func_jax(x):
        # Matrix operations: sum of squares (Frobenius norm squared)
        return jnp.sum(x * x)

    def matrix_func_nabla(x):
        # Matrix operations: sum of squares (Frobenius norm squared)
        return nb.sum(x * x)

    # Test input: 2x3 matrix
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    if JAX_AVAILABLE:
        # JAX computation
        x_jax = jnp.array(x_np)
        jac_jax = jax.jacfwd(matrix_func_jax)(x_jax)
        print(f"JAX jacfwd result: shape {jac_jax.shape}\n{jac_jax}")

    # Nabla computation
    x_nabla = nb.array(x_np)

    try:
        jac_nabla_fn = nb.jacfwd(matrix_func_nabla)
        jac_nabla = jac_nabla_fn(x_nabla)
        print(f"Nabla jacfwd result: shape {jac_nabla.shape}\n{jac_nabla.to_numpy()}")

        if JAX_AVAILABLE:
            np.testing.assert_allclose(jac_nabla.to_numpy(), jac_jax, rtol=1e-6)
            print("✅ Matrix function results match!")

    except Exception as e:
        print(f"❌ Matrix function failed: {e}")
        import traceback

        traceback.print_exc()


def test_nested_jacfwd():
    """Test nested jacfwd calls (Hessian computation)."""
    print("\n=== Nested jacfwd (Hessian) ===")

    def quadratic_func_jax(x):
        term1 = x * x * 2.0
        term2 = x * x * 2.0
        term3 = x * x * 3.0
        result = term1 + term2 + term3
        return result

    def quadratic_func_nabla(x):
        term1 = x * x * 2.0
        term2 = x * x * 2.0
        term3 = x * x * 3.0
        result = term1 + term2 + term3
        return result

    # Test input
    x_np = np.array([1.0, 2.0])

    if JAX_AVAILABLE:
        # JAX nested jacfwd (Hessian)
        x_jax = jnp.array(x_np)

        # First jacobian (gradient)
        grad_fn_jax = jax.jacfwd(quadratic_func_jax)
        grad_jax = grad_fn_jax(x_jax)
        print(f"JAX gradient: shape {grad_jax.shape}, values {grad_jax}")

        # Second jacobian (Hessian)
        hessian_fn_jax = jax.jacfwd(grad_fn_jax)
        hessian_jax = hessian_fn_jax(x_jax)
        print(f"JAX Hessian: shape {hessian_jax.shape}\n{hessian_jax}")

    # Nabla nested jacfwd
    x_nabla = nb.array(x_np)

    try:
        # First jacobian (gradient)
        grad_fn_nabla = nb.jacfwd(quadratic_func_nabla)
        grad_nabla = grad_fn_nabla(x_nabla)
        print(
            f"Nabla gradient: shape {grad_nabla.shape}, values {grad_nabla.to_numpy()}"
        )

        # Second jacobian (Hessian)
        hessian_fn_nabla = nb.jacfwd(grad_fn_nabla)
        hessian_nabla = hessian_fn_nabla(x_nabla)
        print(f"Nabla Hessian: shape {hessian_nabla.shape}\n{hessian_nabla.to_numpy()}")

        if JAX_AVAILABLE:
            np.testing.assert_allclose(grad_nabla.to_numpy(), grad_jax, rtol=1e-6)
            print("✅ Gradient results match!")

            np.testing.assert_allclose(hessian_nabla.to_numpy(), hessian_jax, rtol=1e-6)
            print("✅ Hessian results match!")

    except Exception as e:
        print(f"❌ Nested jacfwd failed: {e}")
        import traceback

        traceback.print_exc()


def test_complex_composition():
    """Test complex function composition using available Nabla operations."""
    print("\n=== Complex Function Composition ===")

    def complex_func_jax(x, y):
        # Complex composition: sin(x*y) + exp(x) * y^2
        return jnp.sin(x * y) + jnp.exp(x) * y**2

    def complex_func_nabla(x, y):
        # Complex composition: sin(x*y) + exp(x) * y^2
        return nb.sin(x * y) + nb.exp(x) * y**2

    # Test inputs
    x_np = np.array([0.5, 1.0])
    y_np = np.array([1.5, 2.0])

    if JAX_AVAILABLE:
        # JAX computation
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)

        jac_jax_both = jax.jacfwd(complex_func_jax, argnums=(0, 1))(x_jax, y_jax)
        print("JAX jacfwd result (both args):")
        for i, jac in enumerate(jac_jax_both):
            print(f"  d/d_arg{i}: shape {jac.shape}\n{jac}")

    # Nabla computation
    x_nabla = nb.array(x_np)
    y_nabla = nb.array(y_np)

    try:
        jac_nabla_fn_both = nb.jacfwd(complex_func_nabla, argnums=(0, 1))
        jac_nabla_both = jac_nabla_fn_both(x_nabla, y_nabla)
        print("Nabla jacfwd result (both args):")
        for i, jac in enumerate(jac_nabla_both):
            print(f"  d/d_arg{i}: shape {jac.shape}\n{jac.to_numpy()}")

        if JAX_AVAILABLE:
            for _i, (jax_jac, nabla_jac) in enumerate(
                zip(jac_jax_both, jac_nabla_both, strict=False)
            ):
                np.testing.assert_allclose(nabla_jac.to_numpy(), jax_jac, rtol=1e-5)
            print("✅ Complex composition results match!")

    except Exception as e:
        print(f"❌ Complex composition failed: {e}")
        import traceback

        traceback.print_exc()


def test_vector_valued_function():
    """Test vector-valued function with vector input."""
    print("\n=== Vector-Valued Function ===")

    def vector_func_jax(x):
        # f: R^3 -> R^2, f(x) = [x[0]*x[1] + x[2]^2, x[0]^2 + x[1]*x[2]]
        return jnp.array([x[0] * x[1] + x[2] ** 2, x[0] ** 2 + x[1] * x[2]])

    def vector_func_nabla(x):
        # f: R^3 -> R^2, f(x) = [x[0]*x[1] + x[2]^2, x[0]^2 + x[1]*x[2]]
        out1 = x[0] * x[1] + x[2] ** 2
        out2 = x[0] ** 2 + x[1] * x[2]
        return nb.concatenate([nb.reshape(out1, (1,)), nb.reshape(out2, (1,))], axis=0)

    # Test input
    x_np = np.array([1.0, 2.0, 3.0])

    if JAX_AVAILABLE:
        # JAX computation
        x_jax = jnp.array(x_np)
        jac_jax = jax.jacfwd(vector_func_jax)(x_jax)
        print(f"JAX jacfwd result: shape {jac_jax.shape}\n{jac_jax}")

    # Nabla computation
    x_nabla = nb.array(x_np)

    try:
        jac_nabla_fn = nb.jacfwd(vector_func_nabla)
        jac_nabla = jac_nabla_fn(x_nabla)
        print(f"Nabla jacfwd result: shape {jac_nabla.shape}\n{jac_nabla.to_numpy()}")

        if JAX_AVAILABLE:
            np.testing.assert_allclose(jac_nabla.to_numpy(), jac_jax, rtol=1e-6)
            print("✅ Vector-valued function results match!")

    except Exception as e:
        print(f"❌ Vector-valued function failed: {e}")


if __name__ == "__main__":
    print("=== ADVANCED JACFWD TESTS ===")

    if JAX_AVAILABLE:
        print("✓ JAX available for ground truth comparison")
    else:
        print("✗ JAX not available - will only test Nabla behavior")

    test_multiple_inputs_multiple_outputs()
    test_matrix_function()
    test_nested_jacfwd()
    test_complex_composition()
    test_vector_valued_function()

    print("\n=== ADVANCED TESTS COMPLETE ===")
