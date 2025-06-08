import nabla as nb
import jax
import jax.numpy as jnp
import numpy as np


# Example function using Nabla's array operations
def foo_nabla(input):
    return nb.sin(input * input)


# Example function using JAX's array operations
def foo_jax(input):
    return jnp.sin(input * input)


# random input for testing
np_input = np.random.randn(2, 3, 4)

# Vectorize, differentiate, accelerate
foo_grads_jax = jax.jit(jax.vmap(jax.jacrev(foo_jax)))
gradients_jax = foo_grads_jax(np_input)
# print("Gradients JAX:", gradients_jax, "Shape:", gradients_jax.shape)

# Vectorize, differentiate, accelerate
# print("XPR:", nb.xpr(nb.jacrev(foo_nabla), nb.array(np_input)))
foo_grads_nabla = nb.jit(nb.vmap(nb.jacrev(foo_nabla)))
gradients_nabla = foo_grads_nabla(nb.array(np_input))


assert jnp.allclose(gradients_jax, gradients_nabla.to_numpy()), (
    "Gradients do not match between JAX and Nabla"
)
print("Gradients match between JAX and Nabla")
