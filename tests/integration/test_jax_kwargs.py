#!/usr/bin/env python3
"""Test JAX vjp behavior with keyword arguments."""

try:
    # We need jax.numpy and the vjp function, but don't directly use the jax module
    import jax.numpy as jnp
    from jax import vjp as jax_vjp

    def test_jax_vjp_keyword_arguments():
        """Test JAX vjp behavior with different argument patterns."""
        print("Testing JAX vjp with keyword arguments...")

        def func_pos(x, y):
            return jnp.sum(x * y)

        def func_kwargs_only(*, x, y):
            return jnp.sum(x * y)

        def func_mixed(a, *, x, y):
            return jnp.sum(a + x * y)

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        a = jnp.array([1.0, 1.0])

        print("\n1. JAX vjp with positional args:")
        try:
            out, vjp_fn = jax_vjp(func_pos, x, y)
            print(f"✓ Success: {out}")
            assert abs(float(out) - 11.0) < 1e-6, "Expected output 11.0"
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise AssertionError(f"Positional args should work: {e}") from e

        print("\n2. JAX vjp with keyword args:")
        try:
            out, vjp_fn = jax_vjp(func_kwargs_only, x=x, y=y)
            print(f"✓ Success: {out}")
            # This is expected to fail according to the original test
        except Exception as e:
            print(f"✗ Failed: {e}")
            # Expected failure, not an assertion error

        print("\n3. JAX vjp with mixed args:")
        try:
            out, vjp_fn = jax_vjp(func_mixed, a, x=x, y=y)
            print(f"✓ Success: {out}")
            # This is expected to fail according to the original test
        except Exception as e:
            print(f"✗ Failed: {e}")
            # Expected failure, not an assertion error

        print("\n4. JAX vjp trying partial/lambda for kwargs:")
        try:
            import functools

            func_partial = functools.partial(func_kwargs_only, x=x, y=y)
            out, vjp_fn = jax_vjp(func_partial)
            print(f"✓ Success with partial: {out}")
            assert abs(float(out) - 11.0) < 1e-6, "Expected output 11.0"
        except Exception as e:
            print(f"✗ Failed with partial: {e}")
            raise AssertionError(f"Partial should work: {e}") from e

        try:

            def func_wrapper():
                return func_kwargs_only(x=x, y=y)

            out, vjp_fn = jax_vjp(func_wrapper)
            print(f"✓ Success with function wrapper: {out}")
            assert abs(float(out) - 11.0) < 1e-6, "Expected output 11.0"
        except Exception as e:
            print(f"✗ Failed with function wrapper: {e}")
            raise AssertionError(f"Function wrapper should work: {e}") from e

except ImportError:

    def test_jax_vjp_keyword_arguments():
        """Test JAX vjp behavior - skipped due to missing JAX."""
        import pytest

        pytest.skip("JAX not available - cannot test JAX behavior")
