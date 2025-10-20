# djit

## Signature

```python
nabla.djit(func: collections.abc.Callable[..., typing.Any] | None = None, show_graph: bool = False, auto_device: bool = True) -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.jit`

## Description

Dynamic JIT compile a function for performance optimization.
This can be used as a function call like `djit(func)` or as a decorator `@djit`.

Args:
    func: Function to optimize with JIT compilation (should take positional arguments)
    show_graph: If True, prints the compiled graph representation when realized.
    auto_device: If True (default) and an accelerator is available, automatically moves CPU-resident input Tensors
        to the default accelerator device before tracing/execution. Unlike static `jit`, dynamic mode does not
        eagerly convert Python scalars to Tensors during the early device pass (to preserve prior semantics).
        Disable by setting to False.

Returns:
    JIT-compiled function with optimized execution

Note:
    This follows JAX's jit API:

    * Only accepts positional arguments
    * For functions requiring keyword arguments, use functools.partial or lambda
    * Supports both list-style (legacy) and unpacked arguments style (JAX-like)
    * Device auto-movement is a Nabla extension controlled by `auto_device`.
