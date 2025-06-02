#!/usr/bin/env python3
"""
Test script to verify that Nabla's VJP transformation matches JAX's behavior
exactly in terms of input/output structure and gradient computation.
"""

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


def test_single_input_single_output():
    """Test VJP with single input, single output."""
    print("Testing single input, single output...")

    # Setup
    x_np = np.array([1.0, 2.0, 3.0])
    x_nb = nb.Array.from_numpy(x_np)
    x_jax = jnp.array(x_np)

    cotangent_np = np.array([1.0, 1.0, 1.0])
    cotangent_nb = nb.Array.from_numpy(cotangent_np)
    cotangent_jax = jnp.array(cotangent_np)

    # Nabla VJP
    def nb_fn(x):
        return nb.sin(x)

    out_nb, vjp_nb = nb.vjp(nb_fn, x_nb)
    grads_nb = vjp_nb(cotangent_nb)

    # JAX VJP
    def jax_fn(x):
        return jnp.sin(x)

    out_jax, vjp_jax = jax.vjp(jax_fn, x_jax)
    grads_jax = vjp_jax(cotangent_jax)

    # Check structure
    print(
        f"  Nabla VJP structure: {type(grads_nb)}, len={len(grads_nb) if hasattr(grads_nb, '__len__') else 'N/A'}"
    )
    print(
        f"  JAX VJP structure: {type(grads_jax)}, len={len(grads_jax) if hasattr(grads_jax, '__len__') else 'N/A'}"
    )

    # Both should return tuples with one element
    assert isinstance(grads_nb, tuple), (
        f"Nabla should return tuple, got {type(grads_nb)}"
    )
    assert isinstance(grads_jax, tuple), (
        f"JAX should return tuple, got {type(grads_jax)}"
    )
    assert len(grads_nb) == 1, (
        f"Nabla should return tuple with 1 element, got {len(grads_nb)}"
    )
    assert len(grads_jax) == 1, (
        f"JAX should return tuple with 1 element, got {len(grads_jax)}"
    )

    # Check values
    assert np.allclose(grads_nb[0].to_numpy(), grads_jax[0]), (
        "Gradient values should match"
    )
    print("  ✓ Single input, single output: PASSED")


def test_multiple_inputs_single_output():
    """Test VJP with multiple inputs, single output."""
    print("Testing multiple inputs, single output...")

    # Setup
    x_np = np.array([1.0, 2.0])
    y_np = np.array([3.0, 4.0])
    x_nb = nb.Array.from_numpy(x_np)
    y_nb = nb.Array.from_numpy(y_np)
    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)

    cotangent_np = np.array([1.0, 1.0])
    cotangent_nb = nb.Array.from_numpy(cotangent_np)
    cotangent_jax = jnp.array(cotangent_np)

    # Nabla VJP
    def nb_fn(x, y):
        return nb.add(x, y)

    out_nb, vjp_nb = nb.vjp(nb_fn, x_nb, y_nb)
    grads_nb = vjp_nb(cotangent_nb)

    # JAX VJP
    def jax_fn(x, y):
        return jnp.add(x, y)

    out_jax, vjp_jax = jax.vjp(jax_fn, x_jax, y_jax)
    grads_jax = vjp_jax(cotangent_jax)

    # Check structure
    print(f"  Nabla VJP structure: {type(grads_nb)}, len={len(grads_nb)}")
    print(f"  JAX VJP structure: {type(grads_jax)}, len={len(grads_jax)}")

    # Both should return tuples with two elements
    assert isinstance(grads_nb, tuple), (
        f"Nabla should return tuple, got {type(grads_nb)}"
    )
    assert isinstance(grads_jax, tuple), (
        f"JAX should return tuple, got {type(grads_jax)}"
    )
    assert len(grads_nb) == 2, (
        f"Nabla should return tuple with 2 elements, got {len(grads_nb)}"
    )
    assert len(grads_jax) == 2, (
        f"JAX should return tuple with 2 elements, got {len(grads_jax)}"
    )

    # Check values
    assert np.allclose(grads_nb[0].to_numpy(), grads_jax[0]), (
        "First gradient should match"
    )
    assert np.allclose(grads_nb[1].to_numpy(), grads_jax[1]), (
        "Second gradient should match"
    )
    print("  ✓ Multiple inputs, single output: PASSED")


def test_nested_structure():
    """Test VJP with nested input structures."""
    print("Testing nested input structures...")

    # Setup
    x_np = np.array([1.0, 2.0])
    y_np = np.array([3.0, 4.0])
    # This dict was created for clarity but isn't used
    # inputs_dict = {"x": x_np, "y": y_np}

    # Convert to Nabla and JAX
    x_nb = nb.Array.from_numpy(x_np)
    y_nb = nb.Array.from_numpy(y_np)
    inputs_nb = {"x": x_nb, "y": y_nb}

    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)
    inputs_jax = {"x": x_jax, "y": y_jax}

    cotangent_np = np.array([1.0, 1.0])
    cotangent_nb = nb.Array.from_numpy(cotangent_np)
    cotangent_jax = jnp.array(cotangent_np)

    # Nabla VJP
    def nb_fn(inputs):
        return nb.mul(inputs["x"], inputs["y"])

    out_nb, vjp_nb = nb.vjp(nb_fn, inputs_nb)
    grads_nb = vjp_nb(cotangent_nb)

    # JAX VJP
    def jax_fn(inputs):
        return jnp.multiply(inputs["x"], inputs["y"])

    out_jax, vjp_jax = jax.vjp(jax_fn, inputs_jax)
    grads_jax = vjp_jax(cotangent_jax)

    # Check structure
    print(f"  Nabla VJP structure: {type(grads_nb)}, len={len(grads_nb)}")
    print(f"  JAX VJP structure: {type(grads_jax)}, len={len(grads_jax)}")

    # Both should return tuples with one element (the dict)
    assert isinstance(grads_nb, tuple), (
        f"Nabla should return tuple, got {type(grads_nb)}"
    )
    assert isinstance(grads_jax, tuple), (
        f"JAX should return tuple, got {type(grads_jax)}"
    )
    assert len(grads_nb) == 1, (
        f"Nabla should return tuple with 1 element, got {len(grads_nb)}"
    )
    assert len(grads_jax) == 1, (
        f"JAX should return tuple with 1 element, got {len(grads_jax)}"
    )

    # Both should have the same dictionary structure
    assert isinstance(grads_nb[0], dict), "Nabla gradient should be a dict"
    assert isinstance(grads_jax[0], dict), "JAX gradient should be a dict"
    assert set(grads_nb[0].keys()) == set(grads_jax[0].keys()), (
        "Gradient dicts should have same keys"
    )

    # Check values
    for key in grads_nb[0]:
        assert np.allclose(grads_nb[0][key].to_numpy(), grads_jax[0][key]), (
            f"Gradient for {key} should match"
        )

    print("  ✓ Nested input structures: PASSED")


def main():
    """Run all compatibility tests."""
    print("=== Nabla VJP Compatibility Test ===")
    print("Verifying that Nabla's VJP matches JAX's behavior exactly\n")

    try:
        test_single_input_single_output()
        test_multiple_inputs_single_output()
        test_nested_structure()

        print("\n🎉 ALL TESTS PASSED!")
        print("Nabla's VJP transformation is fully compatible with JAX!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
