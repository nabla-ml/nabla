# jit

## Signature

```python
nabla.jit(func: 'Callable[..., Any]' = None) -> 'Callable[..., Any]'
```

## Description

Just-in-time compile a function for performance optimization.
This can be used as a function call like `jit(func)` or as a decorator `@jit`.


## Parameters

func: Function to optimize with JIT compilation (should take positional arguments)


## Returns

JIT-compiled function with optimized execution


## Examples

As a function call::

fast_func = jit(my_func)

As a decorator::

@jit
def my_func(x):
return x * 2

## Notes

This follows JAX's jit API:

* Only accepts positional arguments
* For functions requiring keyword arguments, use functools.partial or lambda
* Supports both list-style (legacy) and unpacked arguments style (JAX-like)


## See Also

- {doc}`vmap <core_vmap>` - Vectorization
- {doc}`vjp <core_vjp>`, {doc}`jvp <core_jvp>` - Automatic differentiation

