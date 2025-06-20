#!/usr/bin/env python3

import jax
import jax.numpy as jnp

import nabla as nb


def test_floor_basic():
    """Test basic floor operation"""
    print("Testing basic floor operation...")

    # Test with scalar
    x_nb = nb.array(2.7)
    x_jax = jnp.array(2.7)

    result_nb = nb.floor(x_nb)
    result_jax = jnp.floor(x_jax)

    print(f"Nabla result: {result_nb}")
    print(f"JAX result: {result_jax}")
    print(f"Match: {result_nb.to_numpy() == result_jax}")


def test_floor_jit():
    """Test floor with JIT"""
    print("\nTesting floor with JIT...")

    def f_nb(x):
        return nb.floor(x)

    def f_jax(x):
        return jnp.floor(x)

    x_nb = nb.array(2.7)
    x_jax = jnp.array(2.7)

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


def test_floor_vjp_jit():
    """Test floor with VJP + JIT"""
    print("\nTesting floor with VJP + JIT...")

    def f_nb(x):
        return nb.floor(x)

    def f_jax(x):
        return jnp.floor(x)

    x_nb = nb.array(2.7)
    x_jax = jnp.array(2.7)

    try:
        # Test Nabla JIT(VJP)
        jit_vjp_f_nb = nb.djit(lambda x: nb.vjp(f_nb, x))
        result_nb = jit_vjp_f_nb(x_nb)
        print(f"Nabla JIT(VJP) result: {result_nb}")
    except Exception as e:
        print(f"Nabla JIT(VJP) failed: {e}")

    try:
        # Test JAX JIT(VJP)
        jit_vjp_f_jax = jax.jit(lambda x: jax.vjp(f_jax, x))
        result_jax = jit_vjp_f_jax(x_jax)
        print(f"JAX JIT(VJP) result: {result_jax}")
    except Exception as e:
        print(f"JAX JIT(VJP) failed: {e}")


if __name__ == "__main__":
    test_floor_basic()
    test_floor_jit()
    test_floor_vjp_jit()
