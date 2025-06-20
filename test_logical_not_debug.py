#!/usr/bin/env python3

import jax
import jax.numpy as jnp

import nabla as nb


def test_logical_not_basic():
    """Test basic logical_not operation"""
    print("Testing basic logical_not operation...")

    # Test with scalar
    x_nb = nb.array(True)
    x_jax = jnp.array(True)

    result_nb = nb.logical_not(x_nb)
    result_jax = jnp.logical_not(x_jax)

    print(f"Nabla result: {result_nb}")
    print(f"JAX result: {result_jax}")
    print(f"Match: {result_nb.to_numpy() == result_jax}")


def test_logical_not_array():
    """Test logical_not with array"""
    print("\nTesting logical_not with array...")

    # Test with array
    x_nb = nb.array([True, False, True])
    x_jax = jnp.array([True, False, True])

    result_nb = nb.logical_not(x_nb)
    result_jax = jnp.logical_not(x_jax)

    print(f"Nabla result: {result_nb}")
    print(f"JAX result: {result_jax}")
    print(f"Match: {(result_nb.to_numpy() == result_jax).all()}")


def test_logical_not_jit():
    """Test logical_not with JIT"""
    print("\nTesting logical_not with JIT...")

    def f_nb(x):
        return nb.logical_not(x)

    def f_jax(x):
        return jnp.logical_not(x)

    x_nb = nb.array(True)
    x_jax = jnp.array(True)

    try:
        # Test Nabla JIT
        jit_f_nb = nb.djit(f_nb)
        result_nb = jit_f_nb(x_nb)
        print(f"Nabla JIT result: {result_nb}")
    except Exception as e:
        print(f"Nabla JIT failed: {e}")

    try:
        # Test JAX JIT
        jit_f_jax = jax.jit(f_jax)
        result_jax = jit_f_jax(x_jax)
        print(f"JAX JIT result: {result_jax}")
    except Exception as e:
        print(f"JAX JIT failed: {e}")


if __name__ == "__main__":
    test_logical_not_basic()
    test_logical_not_array()
    test_logical_not_jit()
