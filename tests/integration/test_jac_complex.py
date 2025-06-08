import jax
import jax.numpy as jnp
import numpy as np
import nabla as nb


def flatten_nested(nested):
    if isinstance(nested, (list, tuple)):
        return [item for sublist in nested for item in flatten_nested(sublist)]
    return [nested]


def test_vector_to_scalar_jacobian():
    """Test vector to scalar jacobian against JAX with two inputs and two outputs."""

    # Define functions for both Nabla and JAX
    def func_nb(x, y):
        return (nb.sin(x) * x * y).sum([0, 1]), x * nb.sin(y) * y

    def func_jax(x, y):
        return (jnp.sin(x) * x * y).sum((0, 1)), x * jnp.sin(y) * y

    # Test data - use compatible shapes and values
    x_nb = nb.arange((2, 3))
    y_nb = nb.arange((3,))

    # Convert to JAX arrays with same values
    x_jax = jnp.array(x_nb.to_numpy())
    y_jax = jnp.array(y_nb.to_numpy())

    # Print Nabla XPR
    print("Nabla XPR:")
    print(nb.xpr(func_nb, x_nb, y_nb))

    # Forward pass comparison
    result_nb = func_nb(x_nb, y_nb)
    result_jax = func_jax(x_jax, y_jax)

    # Compare forward pass values
    forward_match_0 = np.allclose(result_nb[0].to_numpy(), result_jax[0])
    forward_match_1 = np.allclose(result_nb[1].to_numpy(), result_jax[1])
    print(f"Forward pass output 0 match: {forward_match_0}")
    print(f"Forward pass output 1 match: {forward_match_1}")

    # FIRST ORDER DERIVATIVES
    print("Nabla Jacobians:")
    jac_fn_nb = nb.jacrev(func_nb, argnums=(0, 1))
    print("Nabla Jacobian XPR:")
    print(nb.xpr(jac_fn_nb, x_nb, y_nb))
    jac_nb = jac_fn_nb(x_nb, y_nb)
    jac_nb_flat = flatten_nested(jac_nb)

    print("JAX Jacobian:")
    jac_fn_jax = jax.jacrev(func_jax, argnums=(0, 1))
    jac_jax = jac_fn_jax(x_jax, y_jax)
    jac_jax_flat = flatten_nested(jac_jax)

    # now we compare the values and the shapes on by one int ehf lattened outptus
    for i, (jac_nb_item, jac_jax_item) in enumerate(zip(jac_nb_flat, jac_jax_flat)):
        match = np.allclose(jac_nb_item.to_numpy(), jac_jax_item)
        print(f"Jacobian output {i} match: {match}")
        # print(f"Nabla shape: {jac_nb_item.shape}, JAX shape: {jac_jax_item.shape}")
        # print(f"Nabla value: {jac_nb_item.to_numpy()}, JAX value: {jac_jax_item}\n")


if __name__ == "__main__":
    test_vector_to_scalar_jacobian()
