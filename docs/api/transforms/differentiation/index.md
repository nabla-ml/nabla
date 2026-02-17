# Differentiation

## `grad`

```python
def grad(fun: 'Callable', argnums: 'int | tuple[int, ...]' = 0, create_graph: 'bool' = False, realize: 'bool' = True) -> 'Callable':
```
Return a function computing the gradient of *fun* (must return a scalar).


---
## `value_and_grad`

```python
def value_and_grad(fun: 'Callable', argnums: 'int | tuple[int, ...]' = 0, create_graph: 'bool' = False, realize: 'bool' = True) -> 'Callable':
```
Return a function computing ``(value, grad)`` of *fun*.


---
## `vjp`

```python
def vjp(fn: 'Callable[..., Any]', *primals: 'Any', has_aux: 'bool' = False) -> 'tuple[Any, Callable[..., tuple[Any, ...]]] | tuple[Any, Callable[..., tuple[Any, ...]], Any]':
```
Compute VJP of *fn* at *primals*. Returns ``(output, vjp_fn[, aux])``.


---
## `jvp`

```python
def jvp(fn: 'Callable[..., Any]', primals: 'tuple[Any, ...]', tangents: 'tuple[Any, ...]', *, has_aux: 'bool' = False) -> 'tuple[Any, Any] | tuple[Any, Any, Any]':
```
Compute JVP of *fn* at *primals* with *tangents*. Returns ``(out, tangent_out[, aux])``.


---
## `jacrev`

```python
def jacrev(fn: 'Callable[..., Any]', argnums: 'int | tuple[int, ...] | list[int] | None' = None, has_aux: 'bool' = False) -> 'Callable[..., Any]':
```
Compute Jacobian of *fn* via reverse-mode (one VJP per output element).


---
## `jacfwd`

```python
def jacfwd(fn: 'Callable[..., Any]', argnums: 'int | tuple[int, ...] | list[int] | None' = None, has_aux: 'bool' = False) -> 'Callable[..., Any]':
```
Compute Jacobian of *fn* via forward-mode (one JVP per input element).


---
