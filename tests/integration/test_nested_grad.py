import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


# Example function using Nabla's array operations
def foo_nabla(input):
    return nb.sum(input * input, axes=0)


# Example function using JAX's array operations
def foo_jax(input):
    return jnp.sum(input * input, axis=0)


# random input for testing
np_input = np.random.randn(2, 3, 4)

# Vectorize, differentiate, accelerate in JAX
foo_grads_jax = jax.jit(jax.vmap(jax.jacrev(foo_jax)))
gradients_jax = foo_grads_jax(np_input)

# Vectorize, differentiate, accelerate in Nabla
foo_grads_nabla = nb.jit(nb.vmap(nb.jacrev(foo_nabla)))
gradients_nabla = foo_grads_nabla(nb.array(np_input))

# Check if gradients match
assert jnp.allclose(gradients_jax, gradients_nabla.to_numpy()), (
    "Gradients do not match between JAX and Nabla"
)
print("Gradients match between JAX and Nabla")
