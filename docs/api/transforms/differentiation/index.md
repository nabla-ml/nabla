# Differentiation

## `grad`

```python
def grad(fun: 'Callable', argnums: 'int | tuple[int, ...]' = 0, create_graph: 'bool' = True, realize: 'bool' = True) -> 'Callable':
```
Return a function computing the gradient of *fun* (must return a scalar).

*create_graph* defaults to ``True`` so that gradients are always
differentiable, enabling higher-order compositions like
``jacrev(grad(f))`` or ``jacfwd(grad(f))`` out of the box.


---
## `value_and_grad`

```python
def value_and_grad(fun: 'Callable', argnums: 'int | tuple[int, ...]' = 0, create_graph: 'bool' = True, realize: 'bool' = True) -> 'Callable':
```
Return a function computing ``(value, grad)`` of *fun*.

See :func:`grad` for *create_graph* semantics.


---
## `vjp`

```python
def vjp(fn: 'Callable[..., Any]', *primals: 'Any', has_aux: 'bool' = False, create_graph: 'bool' = True) -> 'tuple[Any, Callable[..., tuple[Any, ...]]] | tuple[Any, Callable[..., tuple[Any, ...]], Any]':
```
Compute VJP of *fn* at *primals*. Returns ``(output, vjp_fn[, aux])``.

*create_graph* defaults to ``True`` so the returned pullback always
produces differentiable gradients, enabling nested Jacobian compositions.


---
## `jvp`

```python
def jvp(fn: 'Callable[..., Any]', primals: 'tuple[Any, ...]', tangents: 'tuple[Any, ...]', *, has_aux: 'bool' = False, create_graph: 'bool' = True) -> 'tuple[Any, Any] | tuple[Any, Any, Any]':
```
Compute JVP of *fn* at *primals* with *tangents*.

Returns ``(output, tangent_out)`` or ``(output, tangent_out, aux)``
when *has_aux* is True.

Uses a trace-then-forward architecture mirroring VJP's trace-then-backward.


---
## `jacrev`

```python
def jacrev(fn: 'Callable[..., Any]', argnums: 'int | tuple[int, ...] | list[int] | None' = None, has_aux: 'bool' = False) -> 'Callable[..., Any]':
```
Compute Jacobian of *fn* via reverse-mode (``vmap`` over VJP cotangents).


---
## `jacfwd`

```python
def jacfwd(fn: 'Callable[..., Any]', argnums: 'int | tuple[int, ...] | list[int] | None' = None, has_aux: 'bool' = False) -> 'Callable[..., Any]':
```
Compute Jacobian of *fn* via forward-mode (``vmap`` over JVP directions).


---
