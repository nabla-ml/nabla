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

