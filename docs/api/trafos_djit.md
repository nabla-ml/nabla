# djit

## Signature

```python
nabla.djit(func: Optional[collections.abc.Callable[..., Any]] = None, show_graph: bool = False) -> collections.abc.Callable[..., typing.Any]
```

## Description

Dynamic JIT compile a function for performance optimization.
This can be used as a function call like `djit(func)` or as a decorator `@djit`.


## Parameters

func: Function to optimize with JIT compilation (should take positional arguments)


## Returns

JIT-compiled function with optimized execution


## Notes

This follows JAX's jit API:

* Only accepts positional arguments
* For functions requiring keyword arguments, use functools.partial or lambda
* Supports both list-style (legacy) and unpacked arguments style (JAX-like)

