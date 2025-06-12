# Function Transformations

Function transformations for compilation, vectorization, and automatic differentiation.

```{toctree}
:maxdepth: 1

trafos_grad
trafos_jacfwd
trafos_jacrev
trafos_jit
trafos_jvp
trafos_djit
trafos_vjp
trafos_vmap
trafos_xpr
```

## Quick Reference

### `grad`

```python
nabla.grad(fun: 'Callable' = None, argnums: 'int | Sequence[int]' = 0, has_aux: 'bool' = False, holomorphic: 'bool' = False, allow_int: 'bool' = False, reduce_axes: 'Sequence' = (), mode: 'str' = 'reverse') -> 'Callable[..., Any]'
```

Automatic differentiation to compute gradients.

### `jacfwd`

```python
nabla.jacfwd(func: 'Callable[..., Any]', argnums: 'int | tuple[int, ...] | list[int]' = 0, has_aux: 'bool' = False, holomorphic: 'bool' = False, allow_int: 'bool' = False) -> 'Callable[..., Any]'
```

Nabla operation: `jacfwd`

### `jacrev`

```python
nabla.jacrev(func: 'Callable[..., Any]', argnums: 'int | tuple[int, ...] | list[int]' = 0, has_aux: 'bool' = False, holomorphic: 'bool' = False, allow_int: 'bool' = False) -> 'Callable[..., Any]'
```

Nabla operation: `jacrev`

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

### `djit`

```python
nabla.djit(func: 'Callable[..., Any]' = None, show_graph: 'bool' = False) -> 'Callable[..., Any]'
```

Nabla operation: `djit`

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

