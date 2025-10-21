# jit

## Signature

```python
nabla.jit(func: collections.abc.Callable[..., typing.Any] | None = None, static: bool = True, show_graph: bool = False, auto_device: bool = True) -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.jit`

Just-in-time compile a function for performance optimization.
This can be used as a function call like `jit(func)` or as a decorator `@jit`.

Args:
    func: Function to optimize with JIT compilation (should take positional arguments)
    static: If True, compile once and reuse a cached model (fast path). If False, behaves like dynamic JIT (see `djit`).
    show_graph: If True, prints the compiled graph representation when first realized.
    auto_device: If True (default) and an accelerator is available, automatically moves CPU-resident input Tensors
        to the default accelerator device before tracing/execution. In static mode, Python scalars are also
        eagerly converted to device Tensors (since they would be converted during tracing anyway). In dynamic
        mode (`static=False` / `djit`), scalars are left as Python scalars (original behavior) but CPU Tensors
        are still moved. Set to False to disable all automatic device movement/conversion.

Returns:
    JIT-compiled function with optimized execution

Note:
    This follows JAX's jit API:

    * Only accepts positional arguments
    * For functions requiring keyword arguments, use functools.partial or lambda
    * Supports both list-style (legacy) and unpacked arguments style (JAX-like)
    * Device auto-movement is a Nabla extension controlled by `auto_device`.

Example:

    As a function call::
    ```python
    fast_func = jit(my_func)
    ```

    As a decorator::
    ```python
    @jit
    def my_func(x):
        return x * 2
    ```

