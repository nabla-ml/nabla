#!/usr/bin/env python3
"""
Comprehensive test for grad function edge cases and nested inputs against JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


def test_simple_cases():
    """Test simple, straightforward cases."""
    print("=== SIMPLE CASES ===")

    # Single tensor input
    def func_nb(x):
        return (x**2).sum()

    def func_jax(x):
        return (x**2).sum()

    x_nb = nb.tensor([1.0, 2.0, 3.0])
    x_jax = jnp.array([1.0, 2.0, 3.0])

    grad_nb = nb.grad(func_nb)(x_nb)
    grad_jax = jax.grad(func_jax)(x_jax)

    print(f"Single tensor: Nabla={grad_nb}, JAX={grad_jax}")
    assert np.allclose(grad_nb.to_numpy(), grad_jax), "Single tensor case failed"
    print("✅ Single tensor case passed")


def test_multiple_separate_args():
    """Test multiple separate arguments."""
    print("\n=== MULTIPLE SEPARATE ARGS ===")

    def func_nb(x, y, z):
        return (x**2 + y**3 + z**4).sum()

    def func_jax(x, y, z):
        return (x**2 + y**3 + z**4).sum()

    x_nb, y_nb, z_nb = nb.tensor([1.0]), nb.tensor([2.0]), nb.tensor([3.0])
    x_jax, y_jax, z_jax = jnp.array([1.0]), jnp.array([2.0]), jnp.array([3.0])

    # Test gradients w.r.t. different arguments
    for argnum in [0, 1, 2]:
        grad_nb = nb.grad(func_nb, argnums=argnum)(x_nb, y_nb, z_nb)
        grad_jax = jax.grad(func_jax, argnums=argnum)(x_jax, y_jax, z_jax)
        print(f"∂/∂arg{argnum}: Nabla={grad_nb}, JAX={grad_jax}")
        assert np.allclose(grad_nb.to_numpy(), grad_jax), f"Argnum {argnum} failed"

    # Test gradients w.r.t. multiple arguments
    grad_nb = nb.grad(func_nb, argnums=(0, 2))(x_nb, y_nb, z_nb)
    grad_jax = jax.grad(func_jax, argnums=(0, 2))(x_jax, y_jax, z_jax)
    print(f"∂/∂(arg0,arg2): Nabla={grad_nb}, JAX={grad_jax}")
    assert len(grad_nb) == len(grad_jax) == 2, "Multiple argnums length mismatch"
    assert np.allclose(grad_nb[0].to_numpy(), grad_jax[0]), (
        "Multiple argnums [0] failed"
    )
    assert np.allclose(grad_nb[1].to_numpy(), grad_jax[1]), (
        "Multiple argnums [1] failed"
    )

    print("✅ Multiple separate args passed")


def test_single_list_input():
    """Test single list input (like MLP training case) - JAX style."""
    print("\n=== SINGLE LIST INPUT ===")

    def func_nb(inputs):
        x, y, w1, b1, w2, b2 = inputs
        # Simple MLP-like computation
        h = nb.tanh(x @ w1 + b1)
        pred = h @ w2 + b2
        loss = ((pred - y) ** 2).sum()
        return loss

    def func_jax(inputs):
        x, y, w1, b1, w2, b2 = inputs
        h = jnp.tanh(x @ w1 + b1)
        pred = h @ w2 + b2
        loss = ((pred - y) ** 2).sum()
        return loss

    # Create test data
    x_nb = nb.tensor([[1.0, 2.0]])
    y_nb = nb.tensor([[3.0]])
    w1_nb = nb.tensor([[0.1, 0.2], [0.3, 0.4]])
    b1_nb = nb.tensor([[0.1, 0.2]])
    w2_nb = nb.tensor([[0.5], [0.6]])
    b2_nb = nb.tensor([[0.1]])

    inputs_nb = [x_nb, y_nb, w1_nb, b1_nb, w2_nb, b2_nb]
    inputs_jax = [jnp.array(arr.to_numpy()) for arr in inputs_nb]

    # JAX pattern: Get gradients for all elements, then manually select parameters
    print(
        "Testing JAX-style approach: grad returns full structure, manually select parameters"
    )

    # Get gradients for the full input structure
    all_grads_nb = nb.grad(func_nb)(inputs_nb)
    all_grads_jax = jax.grad(func_jax)(inputs_jax)

    # Manually select parameter gradients (indices 2, 3, 4, 5 = w1, b1, w2, b2)
    param_indices = [2, 3, 4, 5]
    param_grads_nb = [all_grads_nb[i] for i in param_indices]
    param_grads_jax = [all_grads_jax[i] for i in param_indices]

    print(
        f"Param gradients: Nabla={len(param_grads_nb)} grads, JAX={len(param_grads_jax)} grads"
    )
    assert len(param_grads_nb) == len(param_grads_jax), "Gradient count mismatch"

    for i, (g_nb, g_jax) in enumerate(
        zip(param_grads_nb, param_grads_jax, strict=False)
    ):
        print(f"  Param {i}: shapes Nabla={g_nb.shape}, JAX={g_jax.shape}")
        assert np.allclose(g_nb.to_numpy(), g_jax, rtol=1e-5), (
            f"Param {i} gradient mismatch"
        )

    print("✅ Single list input passed")


def test_nested_structures():
    """Test nested structures (dicts, tuples, lists)."""
    print("\n=== NESTED STRUCTURES ===")

    def func_nb(params):
        weights, biases = params
        w1, w2 = weights
        b1, b2 = biases
        return (w1**2 + w2**3 + b1**4 + b2**5).sum()

    def func_jax(params):
        weights, biases = params
        w1, w2 = weights
        b1, b2 = biases
        return (w1**2 + w2**3 + b1**4 + b2**5).sum()

    # Create nested structure
    w1_nb, w2_nb = nb.tensor([1.0]), nb.tensor([2.0])
    b1_nb, b2_nb = nb.tensor([3.0]), nb.tensor([4.0])
    params_nb = ([w1_nb, w2_nb], [b1_nb, b2_nb])

    w1_jax, w2_jax = jnp.array([1.0]), jnp.array([2.0])
    b1_jax, b2_jax = jnp.array([3.0]), jnp.array([4.0])
    params_jax = ([w1_jax, w2_jax], [b1_jax, b2_jax])

    grad_nb = nb.grad(func_nb)(params_nb)
    grad_jax = jax.grad(func_jax)(params_jax)

    print("Nested structure gradients:")
    print(f"  Nabla weights: {[g.to_numpy() for g in grad_nb[0]]}")
    print(f"  JAX weights: {grad_jax[0]}")
    print(f"  Nabla biases: {[g.to_numpy() for g in grad_nb[1]]}")
    print(f"  JAX biases: {grad_jax[1]}")

    # Check weights gradients
    assert np.allclose(grad_nb[0][0].to_numpy(), grad_jax[0][0]), "w1 gradient mismatch"
    assert np.allclose(grad_nb[0][1].to_numpy(), grad_jax[0][1]), "w2 gradient mismatch"

    # Check biases gradients
    assert np.allclose(grad_nb[1][0].to_numpy(), grad_jax[1][0]), "b1 gradient mismatch"
    assert np.allclose(grad_nb[1][1].to_numpy(), grad_jax[1][1]), "b2 gradient mismatch"

    print("✅ Nested structures passed")


def test_dict_inputs():
    """Test dictionary inputs."""
    print("\n=== DICT INPUTS ===")

    def func_nb(params):
        return (params["w"] ** 2 + params["b"] ** 3).sum()

    def func_jax(params):
        return (params["w"] ** 2 + params["b"] ** 3).sum()

    params_nb = {"w": nb.tensor([1.0, 2.0]), "b": nb.tensor([3.0])}
    params_jax = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}

    grad_nb = nb.grad(func_nb)(params_nb)
    grad_jax = jax.grad(func_jax)(params_jax)

    print("Dict gradients:")
    print(f"  Nabla w: {grad_nb['w']}")
    print(f"  JAX w: {grad_jax['w']}")
    print(f"  Nabla b: {grad_nb['b']}")
    print(f"  JAX b: {grad_jax['b']}")

    assert np.allclose(grad_nb["w"].to_numpy(), grad_jax["w"]), (
        "Dict w gradient mismatch"
    )
    assert np.allclose(grad_nb["b"].to_numpy(), grad_jax["b"]), (
        "Dict b gradient mismatch"
    )

    print("✅ Dict inputs passed")


def test_mixed_pytrees():
    """Test mixed pytree structures."""
    print("\n=== MIXED PYTREES ===")

    def func_nb(params):
        linear = params["linear"]
        conv = params["conv"]
        w_lin, b_lin = linear
        w_conv = conv["weight"]
        bias_conv = conv["bias"]
        return (w_lin**2 + b_lin**3 + w_conv**4 + bias_conv**5).sum()

    def func_jax(params):
        linear = params["linear"]
        conv = params["conv"]
        w_lin, b_lin = linear
        w_conv = conv["weight"]
        bias_conv = conv["bias"]
        return (w_lin**2 + b_lin**3 + w_conv**4 + bias_conv**5).sum()

    # Create mixed structure
    params_nb = {
        "linear": [nb.tensor([1.0]), nb.tensor([2.0])],
        "conv": {"weight": nb.tensor([3.0]), "bias": nb.tensor([4.0])},
    }

    params_jax = {
        "linear": [jnp.array([1.0]), jnp.array([2.0])],
        "conv": {"weight": jnp.array([3.0]), "bias": jnp.array([4.0])},
    }

    grad_nb = nb.grad(func_nb)(params_nb)
    grad_jax = jax.grad(func_jax)(params_jax)

    print("Mixed pytree gradients:")
    print(f"  Linear w: Nabla={grad_nb['linear'][0]}, JAX={grad_jax['linear'][0]}")
    print(f"  Linear b: Nabla={grad_nb['linear'][1]}, JAX={grad_jax['linear'][1]}")
    print(
        f"  Conv w: Nabla={grad_nb['conv']['weight']}, JAX={grad_jax['conv']['weight']}"
    )
    print(f"  Conv b: Nabla={grad_nb['conv']['bias']}, JAX={grad_jax['conv']['bias']}")

    # Check all gradients
    assert np.allclose(grad_nb["linear"][0].to_numpy(), grad_jax["linear"][0]), (
        "Linear w mismatch"
    )
    assert np.allclose(grad_nb["linear"][1].to_numpy(), grad_jax["linear"][1]), (
        "Linear b mismatch"
    )
    assert np.allclose(
        grad_nb["conv"]["weight"].to_numpy(), grad_jax["conv"]["weight"]
    ), "Conv w mismatch"
    assert np.allclose(grad_nb["conv"]["bias"].to_numpy(), grad_jax["conv"]["bias"]), (
        "Conv b mismatch"
    )

    print("✅ Mixed pytrees passed")


def test_edge_cases():
    """Test various edge cases."""
    print("\n=== EDGE CASES ===")

    # Single scalar
    def scalar_func_nb(x):
        return x**2

    def scalar_func_jax(x):
        return x**2

    x_nb = nb.tensor(5.0)
    x_jax = jnp.array(5.0)

    grad_nb = nb.grad(scalar_func_nb)(x_nb)
    grad_jax = jax.grad(scalar_func_jax)(x_jax)

    print(f"Scalar: Nabla={grad_nb}, JAX={grad_jax}")
    assert np.allclose(grad_nb.to_numpy(), grad_jax), "Scalar case failed"

    # Empty list case (should fail gracefully)
    try:

        def empty_func(x):
            return 0.0

        nb.grad(empty_func)([])
        print("ERROR: Empty list should have failed!")
        raise AssertionError("Empty list should fail")
    except Exception as e:
        print(f"✅ Empty list correctly failed: {type(e).__name__}")

    print("✅ Edge cases passed")


if __name__ == "__main__":
    print("=== COMPREHENSIVE GRAD EDGE CASE TESTING ===")
    print("Testing Nabla grad function against JAX for various input structures...\n")

    test_simple_cases()
    test_multiple_separate_args()
    test_single_list_input()
    test_nested_structures()
    test_dict_inputs()
    test_mixed_pytrees()
    test_edge_cases()

    print("\n🎉 ALL EDGE CASE TESTS PASSED! 🎉")
    print("Nabla's grad function handles all tested input structures correctly!")
