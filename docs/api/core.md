# Core Components

Core components of Endia including the Array class and function transformations.

```{toctree}
:maxdepth: 1
:caption: Functions

core_Array
core_jit
core_jvp
core_vjp
core_vmap
core_xpr
```

## Quick Reference

### {doc}`Array <core_Array>`

```python
endia.Array(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), materialize: 'bool' = False, name: 'str' = '', batch_dims: 'Shape' = ()) -> 'None'
```

The fundamental array type in Endia.

### {doc}`jit <core_jit>`

```python
endia.jit(func: 'Callable[..., Any]' = None) -> 'Callable[..., Any]'
```

Just-in-time compilation for performance optimization.

### {doc}`jvp <core_jvp>`

```python
endia.jvp(func: 'Callable[..., Any]', primals, tangents) -> 'tuple[Any, Any]'
```

Jacobian-vector product for forward-mode automatic differentiation.

### {doc}`vjp <core_vjp>`

```python
endia.vjp(func: 'Callable[..., Any]', *primals) -> 'tuple[Any, Callable]'
```

Vector-Jacobian product for reverse-mode automatic differentiation.

### {doc}`vmap <core_vmap>`

```python
endia.vmap(func=None, in_axes=0, out_axes=0) -> 'Callable[..., Any]'
```

Vectorization transformation for batching operations.

### {doc}`xpr <core_xpr>`

```python
endia.xpr(fn: 'Callable[..., Any]', *primals) -> 'str'
```

Create expression graphs for deferred execution.

