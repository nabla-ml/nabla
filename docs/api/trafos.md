# Function Transformations

Function transformations for compilation, vectorization, and automatic differentiation.

```{toctree}
:maxdepth: 1
:caption: Functions

trafos_djit
trafos_grad
trafos_jacfwd
trafos_jacrev
trafos_jit
trafos_jvp
trafos_vjp
trafos_vmap
trafos_xpr
```

## Quick Reference

### {doc}`djit <trafos_djit>`

```python
nabla.djit(func: Optional[collections.abc.Callable[..., Any]] = None, show_graph: bool = False) -> collections.abc.Callable[..., typing.Any]
```

Nabla operation: `djit`

### {doc}`grad <trafos_grad>`

```python
nabla.grad(fun: collections.abc.Callable | None = None, argnums: int | collections.abc.Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False, reduce_axes: collections.abc.Sequence = (), mode: str = 'reverse') -> collections.abc.Callable[..., typing.Any]
```

Automatic differentiation to compute gradients.

### {doc}`jacfwd <trafos_jacfwd>`

```python
nabla.jacfwd(func: collections.abc.Callable[..., typing.Any], argnums: int | tuple[int, ...] | list[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> collections.abc.Callable[..., typing.Any]
```

Nabla operation: `jacfwd`

### {doc}`jacrev <trafos_jacrev>`

```python
nabla.jacrev(func: collections.abc.Callable[..., typing.Any], argnums: int | tuple[int, ...] | list[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> collections.abc.Callable[..., typing.Any]
```

Nabla operation: `jacrev`

### {doc}`jit <trafos_jit>`

```python
nabla.jit(func: Optional[collections.abc.Callable[..., Any]] = None, static: bool = True, show_graph: bool = False) -> collections.abc.Callable[..., typing.Any]
```

Just-in-time compilation for performance optimization.

### {doc}`jvp <trafos_jvp>`

```python
nabla.jvp(func: collections.abc.Callable[..., typing.Any], primals, tangents, has_aux: bool = False) -> tuple[typing.Any, typing.Any] | tuple[typing.Any, typing.Any, typing.Any]
```

Jacobian-vector product for forward-mode automatic differentiation.

### {doc}`vjp <trafos_vjp>`

```python
nabla.vjp(func: collections.abc.Callable[..., typing.Any], *primals, has_aux: bool = False) -> tuple[typing.Any, collections.abc.Callable] | tuple[typing.Any, collections.abc.Callable, typing.Any]
```

Vector-Jacobian product for reverse-mode automatic differentiation.

### {doc}`vmap <trafos_vmap>`

```python
nabla.vmap(func: collections.abc.Callable | None = None, in_axes: Union[int, NoneType, list, tuple] = 0, out_axes: Union[int, NoneType, list, tuple] = 0) -> collections.abc.Callable[..., typing.Any]
```

Vectorization transformation for batching operations.

### {doc}`xpr <trafos_xpr>`

```python
nabla.xpr(fn: 'Callable[..., Any]', *primals) -> 'str'
```

Create expression graphs for deferred execution.

