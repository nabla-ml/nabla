import numpy as np
import nabla as nb
import jax.numpy as jnp
import jax

def test_max():
    a = nb.arange((2, 3, 4))
    print("input:", a)

    def foo(x):
        return nb.max(x, axes=1)

    # Nabla computations
    value_vjp, vjp_fn = nb.vjp(foo, a)
    print("Nabla VJP Value:", value_vjp)
    grad_vjp = vjp_fn(nb.ones_like(value_vjp))
    print("Nabla VJP Gradient:", grad_vjp)

    value_jvp, grad_jvp = nb.jvp(foo, a, nb.ones_like(a))
    print("Nabla JVP Value:", value_jvp)
    print("Nabla JVP Gradient:", grad_jvp)

    value_jit = nb.jit(foo)(a)
    print("Nabla JIT Value:", value_jit)

    value_vmap = nb.vmap(foo)(a)
    print("Nabla Vmap Value:", value_vmap)

    # JAX comparisons
    a_jax = jnp.array(a.to_numpy())
    
    def foo_jax(x):
        return jnp.max(x, axis=1)

    # JAX VJP
    value_jax_vjp, vjp_fn_jax = jax.vjp(foo_jax, a_jax)
    print("\nJAX VJP Value:", value_jax_vjp)
    grad_jax_vjp = vjp_fn_jax(jnp.ones_like(value_jax_vjp))
    print("JAX VJP Gradient:", grad_jax_vjp[0])  # Note: JAX returns tuple for gradients

    # JAX JVP  
    value_jax_jvp, grad_jax_jvp = jax.jvp(foo_jax, (a_jax,), (jnp.ones_like(a_jax),))
    print("JAX JVP Value:", value_jax_jvp)
    print("JAX JVP Gradient:", grad_jax_jvp)

    # JAX JIT
    value_jax_jit = jax.jit(foo_jax)(a_jax)
    print("JAX JIT Value:", value_jax_jit)

    # JAX Vmap
    value_jax_vmap = jax.vmap(foo_jax)(a_jax)
    print("JAX Vmap Value:", value_jax_vmap)

    # Compare values
    print("\n=== COMPARISONS ===")
    print("VJP Values match:", np.allclose(value_vjp.to_numpy(), value_jax_vjp))
    print("VJP Gradient shapes - Nabla:", grad_vjp.shape, "JAX:", grad_jax_vjp[0].shape)
    print("VJP gradients match:", np.allclose(grad_vjp.to_numpy(), grad_jax_vjp[0]))
    print("JVP Values match:", np.allclose(value_jvp.to_numpy(), value_jax_jvp))
    print("JVP Gradient shapes - Nabla:", grad_jvp.shape, "JAX:", grad_jax_jvp.shape)
    print("JVP gradients match:", np.allclose(grad_jvp.to_numpy(), grad_jax_jvp))
    print("JIT values match:", np.allclose(value_jit.to_numpy(), value_jax_jit))
    print("Vmap values match:", np.allclose(value_vmap.to_numpy(), value_jax_vmap))

def test_argmax():
    a = nb.arange((2, 3, 4))
    print("input:", a)

    def foo(x):
        return nb.argmax(x, axes=-1)

    # Nabla computations
    value_vjp, vjp_fn = nb.vjp(foo, a)
    print("Nabla ArgMax VJP Value:", value_vjp)
    grad_vjp = vjp_fn(nb.ones_like(value_vjp))
    print("Nabla ArgMax VJP Gradient:", grad_vjp)

    value_jvp, grad_jvp = nb.jvp(foo, a, nb.ones_like(a))
    print("Nabla ArgMax JVP Value:", value_jvp)
    print("Nabla ArgMax JVP Gradient:", grad_jvp)

    value_jit = nb.jit(foo)(a)
    print("Nabla ArgMax JIT Value:", value_jit)

    value_vmap = nb.vmap(foo)(a)
    print("Nabla ArgMax Vmap Value:", value_vmap)

    # JAX comparisons
    a_jax = jnp.array(a.to_numpy())
    
    def foo_jax(x):
        return jnp.argmax(x, axis=-1)

    # JAX VJP
    value_jax_vjp, vjp_fn_jax = jax.vjp(foo_jax, a_jax)
    print("\nJAX ArgMax VJP Value:", value_jax_vjp)
    grad_jax_vjp = vjp_fn_jax(jnp.ones_like(value_jax_vjp))
    print("JAX ArgMax VJP Gradient:", grad_jax_vjp[0])  # Note: JAX returns tuple for gradients

    # JAX JVP  
    value_jax_jvp, grad_jax_jvp = jax.jvp(foo_jax, (a_jax,), (jnp.ones_like(a_jax),))
    print("JAX ArgMax JVP Value:", value_jax_jvp)
    print("JAX ArgMax JVP Gradient:", grad_jax_jvp)

    # JAX JIT
    value_jax_jit = jax.jit(foo_jax)(a_jax)
    print("JAX ArgMax JIT Value:", value_jax_jit)

    # JAX Vmap
    value_jax_vmap = jax.vmap(foo_jax)(a_jax)
    print("JAX ArgMax Vmap Value:", value_jax_vmap)

    # Compare values
    print("\n=== COMPARISONS ===")
    print("VJP Values match:", np.array_equal(value_vjp.to_numpy(), value_jax_vjp))
    print("VJP Gradient shapes - Nabla:", grad_vjp.shape, "JAX:", grad_jax_vjp[0].shape)
    print("VJP gradients match (both should be zeros):", np.allclose(grad_vjp.to_numpy(), grad_jax_vjp[0]))
    print("JVP Values match:", np.array_equal(value_jvp.to_numpy(), value_jax_jvp))
    print("JVP Gradient shapes - Nabla:", grad_jvp.shape, "JAX:", grad_jax_jvp.shape)
    # Note: JAX returns weird gradients for argmax JVP, so we just check if Nabla's are zeros
    print("JVP gradients are zeros in Nabla (correct for argmax):", np.allclose(grad_jvp.to_numpy(), 0))
    print("JIT values match:", np.array_equal(value_jit.to_numpy(), value_jax_jit))
    print("Vmap values match:", np.array_equal(value_vmap.to_numpy(), value_jax_vmap))



def test_min_binary():
    a = nb.rand((2, 3, 4))
    b = nb.rand((2, 3, 4))
    print("input a:", a)
    print("input b:", b)

    def foo(x, y):
        return nb.minimum(x, y)
    
    # Nabla computations
    value, vjp_fn = nb.vjp(foo, a, b)
    print("Nabla Min Value:", value)
    grad = vjp_fn(nb.ones_like(value))
    print("Nabla Min Gradient:", grad)

    value, grad = nb.jvp(foo, (a, b), (nb.ones_like(a), nb.ones_like(b)))
    print("Nabla Min JVP Value:", value)
    print("Nabla Min JVP Gradient:", grad)
    
    value_jit = nb.jit(foo)(a, b)
    print("Nabla Min JIT Value:", value_jit)
    
    value_vmap = nb.vmap(foo)(a, b)
    print("Nabla Min Vmap Value:", value_vmap)

    # JAX comparisons
    a_jax = jnp.array(a.to_numpy())
    b_jax = jnp.array(b.to_numpy())
    
    def foo_jax(x, y):
        return jnp.minimum(x, y)

    # JAX VJP
    value_jax, vjp_fn_jax = jax.vjp(foo_jax, a_jax, b_jax)
    print("\nJAX Min Value:", value_jax)
    grad_jax = vjp_fn_jax(jnp.ones_like(value_jax))
    print("JAX Min Gradient:", grad_jax)  # JAX returns tuple for multiple args

    # JAX JVP  
    value_jax_jvp, grad_jax_jvp = jax.jvp(foo_jax, (a_jax, b_jax), (jnp.ones_like(a_jax), jnp.ones_like(b_jax)))
    print("JAX Min JVP Value:", value_jax_jvp)
    print("JAX Min JVP Gradient:", grad_jax_jvp)

    # JAX JIT
    value_jax_jit = jax.jit(foo_jax)(a_jax, b_jax)
    print("JAX Min JIT Value:", value_jax_jit)

    # JAX Vmap
    value_jax_vmap = jax.vmap(foo_jax)(a_jax, b_jax)
    print("JAX Min Vmap Value:", value_jax_vmap)

    # Compare values
    print("\n=== COMPARISONS ===")
    print("Values match:", np.allclose(value.to_numpy(), value_jax))
    print("VJP gradients match:", 
          np.allclose(grad[0].to_numpy(), grad_jax[0]) and 
          np.allclose(grad[1].to_numpy(), grad_jax[1]))
    print("JVP values match:", np.allclose(value.to_numpy(), value_jax_jvp))
    print("JVP gradients match:", np.allclose(grad.to_numpy(), grad_jax_jvp))
    print("JIT values match:", np.allclose(value_jit.to_numpy(), value_jax_jit))
    print("Vmap values match:", np.allclose(value_vmap.to_numpy(), value_jax_vmap))
    
def test_max_binary():
    a = nb.rand((2, 3, 4))
    b = nb.rand((2, 3, 4))
    print("input a:", a)
    print("input b:", b)

    def foo(x, y):
        return nb.maximum(x, y)
    
    # Nabla computations
    value, vjp_fn = nb.vjp(foo, a, b)
    print("Nabla Max Value:", value)
    grad = vjp_fn(nb.ones_like(value))
    print("Nabla Max Gradient:", grad)

    value, grad = nb.jvp(foo, (a, b), (nb.ones_like(a), nb.ones_like(b)))
    print("Nabla Max JVP Value:", value)
    print("Nabla Max JVP Gradient:", grad)
    
    value_jit = nb.jit(foo)(a, b)
    print("Nabla Max JIT Value:", value_jit)
    
    value_vmap = nb.vmap(foo)(a, b)
    print("Nabla Max Vmap Value:", value_vmap)

    # JAX comparisons
    a_jax = jnp.array(a.to_numpy())
    b_jax = jnp.array(b.to_numpy())
    
    def foo_jax(x, y):
        return jnp.maximum(x, y)

    # JAX VJP
    value_jax, vjp_fn_jax = jax.vjp(foo_jax, a_jax, b_jax)
    print("\nJAX Max Value:", value_jax)
    grad_jax = vjp_fn_jax(jnp.ones_like(value_jax))
    print("JAX Max Gradient:", grad_jax)  # JAX returns tuple for multiple args

    # JAX JVP  
    value_jax_jvp, grad_jax_jvp = jax.jvp(foo_jax, (a_jax, b_jax), (jnp.ones_like(a_jax), jnp.ones_like(b_jax)))
    print("JAX Max JVP Value:", value_jax_jvp)
    print("JAX Max JVP Gradient:", grad_jax_jvp)

    # JAX JIT
    value_jax_jit = jax.jit(foo_jax)(a_jax, b_jax)
    print("JAX Max JIT Value:", value_jax_jit)

    # JAX Vmap
    value_jax_vmap = jax.vmap(foo_jax)(a_jax, b_jax)
    print("JAX Max Vmap Value:", value_jax_vmap)

    # Compare values
    print("\n=== COMPARISONS ===")
    print("Values match:", np.allclose(value.to_numpy(), value_jax))
    print("VJP gradients match:", 
          np.allclose(grad[0].to_numpy(), grad_jax[0]) and 
          np.allclose(grad[1].to_numpy(), grad_jax[1]))
    print("JVP values match:", np.allclose(value.to_numpy(), value_jax_jvp))
    print("JVP gradients match:", np.allclose(grad.to_numpy(), grad_jax_jvp))
    print("JIT values match:", np.allclose(value_jit.to_numpy(), value_jax_jit))
    print("Vmap values match:", np.allclose(value_vmap.to_numpy(), value_jax_vmap))

if __name__ == "__main__":
    test_max()
    print("\n" + "="*50)
    test_argmax()
    print("\n" + "="*50)
    test_min_binary()
    print("\n" + "="*50)
    test_max_binary()
