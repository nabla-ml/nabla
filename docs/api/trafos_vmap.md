# vmap

## Signature

```python
nabla.vmap(func: collections.abc.Callable | None = None, in_axes: Union[int, NoneType, list, tuple] = 0, out_axes: Union[int, NoneType, list, tuple] = 0) -> collections.abc.Callable[..., typing.Any]
```

## Description

Creates a function that maps a function over axes of pytrees.

`vmap` is a transformation that converts a function designed for single
data points into a function that can operate on batches of data points.
It achieves this by adding a batch dimension to all operations within
the function, enabling efficient, parallel execution.


## Parameters

func: The function to be vectorized. It should be written as if it
operates on a single example.
in_axes: Specifies which axis of the input(s) to map over. Can be an
integer, None, or a pytree of these values. `None` indicates
that the corresponding input should be broadcast.
out_axes: Specifies where to place the batch axis in the output(s).


## Returns

A vectorized function with the same input/output structure as `func`.

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

- {doc}`jit <core_jit>` - Just-in-time compilation
- {doc}`vjp <core_vjp>`, {doc}`jvp <core_jvp>` - Automatic differentiation

