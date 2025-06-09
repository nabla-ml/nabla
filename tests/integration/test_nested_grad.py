import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


def test_nested_grad_jit_vmap_jacrev():
    """Test nested transformations: jit(vmap(jacrev(func))) against JAX."""

    # Example function using Nabla's array operations
    def foo_nabla(input):
        return nb.sum(input * input, axes=0)

    # Example function using JAX's array operations
    def foo_jax(input):
        return jnp.sum(input * input, axis=0)

    # Fixed seed for reproducible testing
    np.random.seed(42)
    np_input = np.random.randn(2, 3, 4)

    # Vectorize, differentiate, accelerate in JAX
    foo_grads_jax = jax.jit(jax.vmap(jax.jacrev(foo_jax)))
    gradients_jax = foo_grads_jax(np_input)

    # Vectorize, differentiate, accelerate in Nabla
    foo_grads_nabla = nb.jit(nb.vmap(nb.jacrev(foo_nabla)))
    gradients_nabla = foo_grads_nabla(nb.array(np_input))

    # Check if gradients match
    assert jnp.allclose(
        gradients_jax, gradients_nabla.to_numpy(), rtol=1e-5, atol=1e-6
    ), "Gradients do not match between JAX and Nabla"
    print("✓ Nested grad test: jit(vmap(jacrev(func))) passed")
