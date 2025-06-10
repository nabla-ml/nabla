# Core Components

Core components of Nabla including the Array class and function transformations.

```{toctree}
:maxdepth: 1

core_Array
core_jit
core_jvp
core_vjp
core_vmap
core_xpr
```

## Quick Reference

### `Array`

```python
nabla.Array(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), materialize: 'bool' = False, name: 'str' = '', batch_dims: 'Shape' = ()) -> 'None'
```

The fundamental array type in Nabla.

### `jit`

```python
nabla.jit(func: 'Callable[..., Any]' = None, static: 'bool' = False, show_graph: 'bool' = False) -> 'Callable[..., Any]'
```

Just-in-time compilation for performance optimization.

### `jvp`

```python
nabla.jvp(func: 'Callable[..., Any]', primals, tangents, has_aux: 'bool' = False) -> 'tuple[Any, Any] | tuple[Any, Any, Any]'
```

Jacobian-vector product for forward-mode automatic differentiation.

### `vjp`

```python
nabla.vjp(func: 'Callable[..., Any]', *primals, has_aux: 'bool' = False) -> 'tuple[Any, Callable]'
```

Vector-Jacobian product for reverse-mode automatic differentiation.

### `vmap`

```python
nabla.vmap(func=None, in_axes=0, out_axes=0) -> 'Callable[..., Any]'
```

Vectorization transformation for batching operations.

### `xpr`

```python
nabla.xpr(fn: 'Callable[..., Any]', *primals) -> 'str'
```

Create expression graphs for deferred execution.

