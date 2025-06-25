import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


def test_grad_reverse_mode():
    """Test grad function in reverse mode against JAX."""

    # Define scalar functions for both Nabla and JAX
    def func_nb(x):
        return (nb.sin(x) * x).sum()  # Sum to make it scalar

    def func_jax(x):
        return (jnp.sin(x) * x).sum()  # Sum to make it scalar

    # Test data - use compatible shapes and values
    x_nb = nb.ndarange((2, 3))
    x_jax = jnp.array(x_nb.to_numpy())

    print("=== REVERSE MODE GRAD TEST ===")
    print(f"x_nb shape: {x_nb.shape}, x_jax shape: {x_jax.shape}")

    # Forward pass comparison
    print("\n=== FORWARD PASS COMPARISON ===")
    result_nb = func_nb(x_nb)
    result_jax = func_jax(x_jax)
    print(f"Nabla result shape: {result_nb.shape}")
    print(f"JAX result shape: {result_jax.shape}")
    print(f"Forward values match: {np.allclose(result_nb.to_numpy(), result_jax)}")

    # First derivative (Gradient) comparison - REVERSE MODE
    print("\n=== FIRST DERIVATIVE (GRADIENT) COMPARISON - REVERSE MODE ===")
    grad_fn_nb = nb.grad(func_nb, mode="reverse")  # Use reverse mode
    grad_fn_jax = jax.grad(func_jax)  # JAX grad defaults to reverse mode

    grad_nb = grad_fn_nb(x_nb)
    grad_jax = grad_fn_jax(x_jax)

    print("Nabla XPR:")
    print(nb.xpr(grad_fn_nb, x_nb))
    print(f"Nabla gradient shape: {grad_nb.shape}")
    print(f"JAX gradient shape: {grad_jax.shape}")
    print(
        f"Gradient values match: {np.allclose(grad_nb.to_numpy(), grad_jax, rtol=1e-5, atol=1e-6)}"
    )

    if not np.allclose(grad_nb.to_numpy(), grad_jax, rtol=1e-5, atol=1e-6):
        print("WARNING: Gradient values don't match!")
        max_diff = np.max(np.abs(grad_nb.to_numpy() - grad_jax))
        print(f"Max difference: {max_diff}")

    # Second derivative (Hessian) comparison - use jacrev for higher order
    print("\n=== SECOND DERIVATIVE (HESSIAN) COMPARISON ===")
    hess_fn_nb = nb.jacrev(grad_fn_nb)  # Use jacrev for higher order
    hess_fn_jax = jax.jacrev(grad_fn_jax)

    hess_nb = hess_fn_nb(x_nb)
    hess_jax = hess_fn_jax(x_jax)

    print("Nabla XPR:")
    print(nb.xpr(hess_fn_nb, x_nb))
    print(f"Nabla hessian shape: {hess_nb.shape}")
    print(f"JAX hessian shape: {hess_jax.shape}")
    print(
        f"Hessian values match: {np.allclose(hess_nb.to_numpy(), hess_jax, rtol=1e-4, atol=1e-5)}"
    )

    # Final assertions for test validation
    assert np.allclose(result_nb.to_numpy(), result_jax), "Forward pass must match"
    assert np.allclose(grad_nb.to_numpy(), grad_jax, rtol=1e-5, atol=1e-6), (
        "Gradient must match"
    )
    assert np.allclose(hess_nb.to_numpy(), hess_jax, rtol=1e-4, atol=1e-5), (
        "Hessian must match"
    )

    print("\n✅ Reverse mode grad tests passed!")


def test_grad_forward_mode():
    """Test grad function in forward mode against JAX jacfwd."""

    # Define scalar functions for both Nabla and JAX
    def func_nb(x):
        return (nb.sin(x) * x).sum()  # Sum to make it scalar

    def func_jax(x):
        return (jnp.sin(x) * x).sum()  # Sum to make it scalar

    # Test data
    x_nb = nb.ndarange((2, 3))
    x_jax = jnp.array(x_nb.to_numpy())

    print("\n\n=== FORWARD MODE GRAD TEST ===")
    print(f"x_nb shape: {x_nb.shape}, x_jax shape: {x_jax.shape}")

    # First derivative (Gradient) comparison - FORWARD MODE
    print("\n=== FIRST DERIVATIVE (GRADIENT) COMPARISON - FORWARD MODE ===")
    grad_fn_nb = nb.grad(func_nb, mode="forward")  # Use forward mode
    grad_fn_jax = jax.jacfwd(func_jax)  # JAX jacfwd for forward mode

    grad_nb = grad_fn_nb(x_nb)
    grad_jax = grad_fn_jax(x_jax)

    print("Nabla XPR:")
    print(nb.xpr(grad_fn_nb, x_nb))
    print(f"Nabla gradient shape: {grad_nb.shape}")
    print(f"JAX gradient shape: {grad_jax.shape}")
    print(
        f"Gradient values match: {np.allclose(grad_nb.to_numpy(), grad_jax, rtol=1e-5, atol=1e-6)}"
    )

    if not np.allclose(grad_nb.to_numpy(), grad_jax, rtol=1e-5, atol=1e-6):
        print("WARNING: Gradient values don't match!")
        max_diff = np.max(np.abs(grad_nb.to_numpy() - grad_jax))
        print(f"Max difference: {max_diff}")

    # Second derivative (Hessian) comparison - use jacfwd for higher order
    print("\n=== SECOND DERIVATIVE (HESSIAN) COMPARISON ===")
    hess_fn_nb = nb.jacfwd(grad_fn_nb)  # Use jacfwd for higher order
    hess_fn_jax = jax.jacfwd(grad_fn_jax)

    hess_nb = hess_fn_nb(x_nb)
    hess_jax = hess_fn_jax(x_jax)

    print("Nabla XPR:")
    print(nb.xpr(hess_fn_nb, x_nb))
    print(f"Nabla hessian shape: {hess_nb.shape}")
    print(f"JAX hessian shape: {hess_jax.shape}")
    print(
        f"Hessian values match: {np.allclose(hess_nb.to_numpy(), hess_jax, rtol=1e-4, atol=1e-5)}"
    )

    # Final assertions
    assert np.allclose(grad_nb.to_numpy(), grad_jax, rtol=1e-5, atol=1e-6), (
        "Forward mode gradient must match"
    )
    assert np.allclose(hess_nb.to_numpy(), hess_jax, rtol=1e-4, atol=1e-5), (
        "Forward mode hessian must match"
    )

    print("\n✅ Forward mode grad tests passed!")


def test_grad_decorator_style():
    """Test grad function used as a decorator."""

    print("\n\n=== DECORATOR STYLE TEST ===")

    # Test decorator with reverse mode (default)
    @nb.grad
    def loss_func_nb(x):
        return (x**2).sum()

    @jax.grad
    def loss_func_jax(x):
        return (x**2).sum()

    x_nb = nb.array([1.0, 2.0, 3.0])
    x_jax = jnp.array([1.0, 2.0, 3.0])

    grad_nb = loss_func_nb(x_nb)
    grad_jax = loss_func_jax(x_jax)

    print(f"Nabla decorator gradient: {grad_nb}")
    print(f"JAX decorator gradient: {grad_jax}")
    print(f"Decorator results match: {np.allclose(grad_nb.to_numpy(), grad_jax)}")

    # Test decorator with forward mode
    @nb.grad(mode="forward")
    def loss_func_nb_fwd(x):
        return (x**3).sum()

    # JAX equivalent using jacfwd
    def loss_func_jax_fwd(x):
        return (x**3).sum()

    grad_fn_jax_fwd = jax.jacfwd(loss_func_jax_fwd)

    grad_nb_fwd = loss_func_nb_fwd(x_nb)
    grad_jax_fwd = grad_fn_jax_fwd(x_jax)

    print(f"Nabla forward decorator gradient: {grad_nb_fwd}")
    print(f"JAX forward gradient: {grad_jax_fwd}")
    print(
        f"Forward decorator results match: {np.allclose(grad_nb_fwd.to_numpy(), grad_jax_fwd)}"
    )

    assert np.allclose(grad_nb.to_numpy(), grad_jax), "Reverse decorator must match"
    assert np.allclose(grad_nb_fwd.to_numpy(), grad_jax_fwd), (
        "Forward decorator must match"
    )

    print("\n✅ Decorator style tests passed!")


def test_grad_multi_argument():
    """Test grad function with multiple arguments using argnums."""

    print("\n\n=== MULTI-ARGUMENT TEST ===")

    def sum_squares_nb(x, y):
        return (x**2 + y**2).sum()

    def sum_squares_jax(x, y):
        return (x**2 + y**2).sum()

    x_nb, y_nb = nb.array([1.0, 2.0]), nb.array([3.0, 4.0])
    x_jax, y_jax = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])

    # Test gradient w.r.t. first argument
    grad_wrt_x_nb = nb.grad(sum_squares_nb, argnums=0)
    grad_wrt_x_jax = jax.grad(sum_squares_jax, argnums=0)

    grad_x_nb = grad_wrt_x_nb(x_nb, y_nb)
    grad_x_jax = grad_wrt_x_jax(x_jax, y_jax)

    print(f"∂/∂x Nabla: {grad_x_nb}")
    print(f"∂/∂x JAX: {grad_x_jax}")
    print(f"∂/∂x match: {np.allclose(grad_x_nb.to_numpy(), grad_x_jax)}")

    # Test gradient w.r.t. second argument
    grad_wrt_y_nb = nb.grad(sum_squares_nb, argnums=1)
    grad_wrt_y_jax = jax.grad(sum_squares_jax, argnums=1)

    grad_y_nb = grad_wrt_y_nb(x_nb, y_nb)
    grad_y_jax = grad_wrt_y_jax(x_jax, y_jax)

    print(f"∂/∂y Nabla: {grad_y_nb}")
    print(f"∂/∂y JAX: {grad_y_jax}")
    print(f"∂/∂y match: {np.allclose(grad_y_nb.to_numpy(), grad_y_jax)}")

    # Test gradient w.r.t. both arguments
    grad_wrt_both_nb = nb.grad(sum_squares_nb, argnums=(0, 1))
    grad_wrt_both_jax = jax.grad(sum_squares_jax, argnums=(0, 1))

    grad_both_nb = grad_wrt_both_nb(x_nb, y_nb)
    grad_both_jax = grad_wrt_both_jax(x_jax, y_jax)

    print(f"∇ Nabla: {grad_both_nb}")
    print(f"∇ JAX: {grad_both_jax}")
    grad_x_match = np.allclose(grad_both_nb[0].to_numpy(), grad_both_jax[0])
    grad_y_match = np.allclose(grad_both_nb[1].to_numpy(), grad_both_jax[1])
    print(f"∇ match: {grad_x_match and grad_y_match}")

    assert np.allclose(grad_x_nb.to_numpy(), grad_x_jax), "∂/∂x must match"
    assert np.allclose(grad_y_nb.to_numpy(), grad_y_jax), "∂/∂y must match"
    assert grad_x_match and grad_y_match, "∇ must match"

    print("\n✅ Multi-argument tests passed!")


def test_grad_with_aux():
    """Test grad function with auxiliary data."""

    print("\n\n=== HAS_AUX TEST ===")

    def func_with_aux_nb(x):
        main_output = (x**2).sum()
        aux_data = {"debug": "auxiliary data", "norm": nb.sqrt((x**2).sum())}
        return main_output, aux_data

    def func_with_aux_jax(x):
        main_output = (x**2).sum()
        aux_data = {"debug": "auxiliary data", "norm": jnp.sqrt((x**2).sum())}
        return main_output, aux_data

    x_nb = nb.array([3.0, 4.0])
    x_jax = jnp.array([3.0, 4.0])

    # Test gradient with auxiliary data
    grad_with_aux_nb = nb.grad(func_with_aux_nb, has_aux=True)
    grad_with_aux_jax = jax.grad(func_with_aux_jax, has_aux=True)

    grad_nb, aux_nb = grad_with_aux_nb(x_nb)
    grad_jax, aux_jax = grad_with_aux_jax(x_jax)

    print(f"Nabla gradient: {grad_nb}")
    print(f"JAX gradient: {grad_jax}")
    print(f"Nabla aux: {aux_nb}")
    print(f"JAX aux: {aux_jax}")
    print(f"Gradient match: {np.allclose(grad_nb.to_numpy(), grad_jax)}")
    print(f"Aux debug match: {aux_nb['debug'] == aux_jax['debug']}")

    # Check auxiliary norm values match
    aux_norm_match = np.allclose(aux_nb["norm"].to_numpy(), aux_jax["norm"])
    print(f"Aux norm match: {aux_norm_match}")

    assert np.allclose(grad_nb.to_numpy(), grad_jax), "Gradient with aux must match"
    assert aux_nb["debug"] == aux_jax["debug"], "Aux debug must match"
    assert aux_norm_match, "Aux norm must match"

    print("\n✅ Has_aux tests passed!")


def test_grad_scalar_validation():
    """Test that grad function properly validates scalar outputs."""

    print("\n\n=== SCALAR VALIDATION TEST ===")

    # Define a function that returns a vector (should fail)
    def vector_func(x):
        return x * 2  # Returns a vector, not a scalar

    x_nb = nb.array([1.0, 2.0, 3.0])

    print("Testing vector function (should raise ValueError)...")

    try:
        grad_fn = nb.grad(vector_func)
        grad_fn(x_nb)  # This should raise an error
        print("ERROR: Should have raised ValueError for non-scalar output!")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
        assert "Gradient only defined for scalar-output functions" in str(e)

    # Test that JAX also raises the same error
    def vector_func_jax(x):
        return x * 2

    x_jax = jnp.array([1.0, 2.0, 3.0])

    print("Testing JAX vector function (should also raise ValueError)...")

    try:
        grad_fn_jax = jax.grad(vector_func_jax)
        grad_fn_jax(x_jax)  # This should also raise an error
        print("ERROR: JAX should have raised ValueError for non-scalar output!")
        raise AssertionError("JAX should have raised ValueError")
    except Exception as e:
        print(f"✅ JAX also correctly raised error: {type(e).__name__}: {e}")

    print("\n✅ Scalar validation tests passed!")


if __name__ == "__main__":
    print("=== TESTING NABLA'S GRAD FUNCTION ===")
    print("Testing both reverse-mode and forward-mode against JAX...")

    # Test reverse mode (default)
    test_grad_reverse_mode()

    # Test forward mode
    test_grad_forward_mode()

    # Test additional features
    test_grad_decorator_style()
    test_grad_multi_argument()
    test_grad_with_aux()
    test_grad_scalar_validation()

    print("\n🎉 ALL GRAD TESTS PASSED! 🎉")
    print("Nabla's grad function works perfectly in both reverse and forward modes!")
