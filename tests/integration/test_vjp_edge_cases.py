#!/usr/bin/env python3
"""
Additional edge case tests for Nabla VJP compatibility.
"""

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


def test_matrix_operations():
    """Test VJP with matrix operations."""
    print("Testing matrix operations...")

    # Setup matrices
    A_np = np.random.randn(3, 4)
    B_np = np.random.randn(4, 2)
    A_nb = nb.Array.from_numpy(A_np)
    B_nb = nb.Array.from_numpy(B_np)
    A_jax = jnp.array(A_np)
    B_jax = jnp.array(B_np)

    # Cotangent for result (3, 2)
    cotangent_np = np.random.randn(3, 2)
    cotangent_nb = nb.Array.from_numpy(cotangent_np)
    cotangent_jax = jnp.array(cotangent_np)

    # Nabla VJP
    def nb_fn(A, B):
        return nb.matmul(A, B)

    out_nb, vjp_nb = nb.vjp(nb_fn, A_nb, B_nb)
    grads_nb = vjp_nb(cotangent_nb)

    # JAX VJP
    def jax_fn(A, B):
        return jnp.matmul(A, B)

    out_jax, vjp_jax = jax.vjp(jax_fn, A_jax, B_jax)
    grads_jax = vjp_jax(cotangent_jax)

    # Check structure and values
    assert len(grads_nb) == len(grads_jax) == 2
    assert np.allclose(grads_nb[0].to_numpy(), grads_jax[0], rtol=1e-5)
    assert np.allclose(grads_nb[1].to_numpy(), grads_jax[1], rtol=1e-5)
    print("  ✓ Matrix operations: PASSED")


def test_complex_function():
    """Test VJP with a more complex function."""
    print("Testing complex function...")

    # Setup
    x_np = np.array([0.5, 1.0, 1.5])
    x_nb = nb.Array.from_numpy(x_np)
    x_jax = jnp.array(x_np)

    cotangent_np = 1.0  # Scalar output
    cotangent_nb = nb.Array.from_numpy(np.array(cotangent_np))
    cotangent_jax = jnp.array(cotangent_np)

    # Nabla VJP - complex function: sum(sin(x) * exp(x))
    def nb_fn(x):
        return nb.sum(nb.mul(nb.sin(x), nb.exp(x)))

    out_nb, vjp_nb = nb.vjp(nb_fn, x_nb)
    grads_nb = vjp_nb(cotangent_nb)

    # JAX VJP
    def jax_fn(x):
        return jnp.sum(jnp.sin(x) * jnp.exp(x))

    out_jax, vjp_jax = jax.vjp(jax_fn, x_jax)
    grads_jax = vjp_jax(cotangent_jax)

    # Check structure and values
    assert len(grads_nb) == len(grads_jax) == 1
    assert np.allclose(grads_nb[0].to_numpy(), grads_jax[0], rtol=1e-5)
    print("  ✓ Complex function: PASSED")


def test_list_inputs():
    """Test VJP with list inputs."""
    print("Testing list inputs...")

    # Setup
    x_np = np.array([1.0, 2.0])
    y_np = np.array([3.0, 4.0])
    z_np = np.array([5.0, 6.0])

    inputs_nb = [
        nb.Array.from_numpy(x_np),
        nb.Array.from_numpy(y_np),
        nb.Array.from_numpy(z_np),
    ]
    inputs_jax = [jnp.array(x_np), jnp.array(y_np), jnp.array(z_np)]

    cotangent_np = np.array([1.0, 1.0])
    cotangent_nb = nb.Array.from_numpy(cotangent_np)
    cotangent_jax = jnp.array(cotangent_np)

    # Nabla VJP
    def nb_fn(inputs):
        return nb.add(nb.add(inputs[0], inputs[1]), inputs[2])

    out_nb, vjp_nb = nb.vjp(nb_fn, inputs_nb)
    grads_nb = vjp_nb(cotangent_nb)

    # JAX VJP
    def jax_fn(inputs):
        return jnp.add(jnp.add(inputs[0], inputs[1]), inputs[2])

    out_jax, vjp_jax = jax.vjp(jax_fn, inputs_jax)
    grads_jax = vjp_jax(cotangent_jax)

    # Check structure and values
    assert len(grads_nb) == len(grads_jax) == 1
    assert isinstance(grads_nb[0], list)
    assert isinstance(grads_jax[0], list)
    assert len(grads_nb[0]) == len(grads_jax[0]) == 3

    for i in range(3):
        assert np.allclose(grads_nb[0][i].to_numpy(), grads_jax[0][i], rtol=1e-5)

    print("  ✓ List inputs: PASSED")


def main():
    """Run edge case tests."""
    print("=== Nabla VJP Edge Case Tests ===")
    print("Testing additional VJP scenarios\n")

    try:
        test_matrix_operations()
        test_complex_function()
        test_list_inputs()

        print("\n🎉 ALL EDGE CASE TESTS PASSED!")
        print("Nabla's VJP is robust and fully JAX-compatible!")

    except Exception as e:
        print(f"\n❌ EDGE CASE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
