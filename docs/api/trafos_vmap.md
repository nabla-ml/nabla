# vmap

## Signature

```python
nabla.vmap(func=None, in_axes=0, out_axes=0) -> 'Callable[..., Any]'
```

## Description

Enhanced vmap with clean pytree support.

This is a simplified, clean implementation that supports all JAX vmap features:
- Pytree inputs/outputs with matching axis specifications
- Broadcasting (axis=None) and batching (axis=int)
- Nested structures (tuples, lists, dicts)
- Both list-style and unpacked argument calling conventions

## Examples

```python
import nabla as nb

# Vectorize function over batch dimension
def dot_product(a, b):
    return nb.sum(a * b)

# Vectorize over first dimension
batch_dot = nb.vmap(dot_product, in_axes=(0, 0))

a_batch = nb.randn((10, 5))  # 10 vectors of length 5
b_batch = nb.randn((10, 5))
results = batch_dot(a_batch, b_batch)  # 10 dot products
```

## See Also

- {doc}`jit <trafos_jit>` - Just-in-time compilation
- {doc}`vjp <trafos_vjp>`, {doc}`jvp <trafos_jvp>` - Automatic differentiation

