# xpr

## Signature

```python
nabla.xpr(fn: 'Callable[..., Any]', *primals) -> 'str'
```

## Description

Get a JAX-like string representation of the function's computation graph.

## Parameters

fn: Function to trace (should take positional arguments)
*primals: Positional arguments to the function (can be arbitrary pytrees)

## Returns

JAX-like string representation of the computation graph

## Notes

This follows the same flexible API as vjp, jvp, and vmap:

- Accepts functions with any number of positional arguments
- For functions requiring keyword arguments, use functools.partial or lambda

## Examples

```python
import nabla as nb

# Single input function
def f(x):
    return x ** 2

x = nb.array([1.0, 2.0])
print(nb.xpr(f, x))

# Multiple input function  
def g(x, y):
    return x * y + x ** 2

x = nb.array([1.0, 2.0])
y = nb.array([3.0, 4.0])
print(nb.xpr(g, x, y))
```

## See Also

- {doc}`vjp <core_vjp>` - Vector-Jacobian product
- {doc}`jvp <core_jvp>` - Jacobian-vector product
