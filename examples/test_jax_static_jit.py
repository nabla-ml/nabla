#!/usr/bin/env python3
"""Test how JAX handles static JIT with changing parameters."""

import jax.numpy as jnp
from jax import jit


def adam_update_jax(param, grad, m, v, step):
    """Adam update that uses step parameter."""
    beta1, beta2, lr, eps = 0.9, 0.999, 0.001, 1e-8

    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

    # This line uses step in computation
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step

    new_param = param - lr * (new_m / bias_correction1) / (
        (new_v / bias_correction2) ** 0.5 + eps
    )

    return new_param, new_m, new_v


# Regular JIT (should work fine)
adam_update_regular_jit = jit(adam_update_jax)

# Static JIT (let's see what happens)
adam_update_static_jit = jit(adam_update_jax, static_argnums=(4,))  # step is static


def test_jax_static_behavior():
    """Test how JAX handles static JIT with changing step."""
    print("Testing JAX static JIT behavior...")

    # Initialize test data
    param = jnp.array([[1.0, 2.0]])
    grad = jnp.array([[0.1, 0.2]])
    m = jnp.zeros_like(param)
    v = jnp.zeros_like(param)

    print("Initial param:", param)

    # Test regular JIT
    print("\n=== Regular JIT ===")
    param_jit, m_jit, v_jit = param, m, v
    for step in range(1, 4):
        param_jit, m_jit, v_jit = adam_update_regular_jit(
            param_jit, grad, m_jit, v_jit, step
        )
        print(f"Step {step} - param: {param_jit}")

    # Test static JIT with step as static argument
    print("\n=== Static JIT (step as static) ===")
    param_static, m_static, v_static = param, m, v
    for step in range(1, 4):
        param_static, m_static, v_static = adam_update_static_jit(
            param_static, grad, m_static, v_static, step
        )
        print(f"Step {step} - param: {param_static}")


if __name__ == "__main__":
    test_jax_static_behavior()
