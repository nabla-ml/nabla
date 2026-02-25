# Differentiation

## `grad`

```python
def grad(fun: 'Callable', argnums: 'int | tuple[int, ...]' = 0, create_graph: 'bool' = True, realize: 'bool' = True) -> 'Callable':
```
Return a function that computes the gradient of *fun*.

*fun* must return a scalar tensor. The returned callable accepts the same
arguments as *fun* and returns the gradient with respect to the inputs
specified by *argnums*.

**Parameters**

- **`fun`** – Scalar-valued function to differentiate.
- **`argnums`** – Index or tuple of indices of positional arguments to
differentiate with respect to. Default: ``0`` (first argument).
- **`create_graph`** – If ``True`` (default), the gradient is itself
differentiable, enabling higher-order derivatives such as
``jacrev(grad(f))``.
- **`realize`** – If ``True`` and *create_graph* is ``False``, eagerly
materialise the gradient tensors before returning.

**Returns**

A callable with the same signature as *fun* that returns the gradient
(or a tuple of gradients when *argnums* is a tuple).


---
## `value_and_grad`

```python
def value_and_grad(fun: 'Callable', argnums: 'int | tuple[int, ...]' = 0, create_graph: 'bool' = True, realize: 'bool' = True) -> 'Callable':
```
Return a function that evaluates *fun* and its gradient simultaneously.

More efficient than calling *fun* and :func:`grad` separately because
the forward pass is shared.

**Parameters**

- **`fun`** – Scalar-valued function to differentiate.
- **`argnums`** – Index or tuple of indices of positional arguments to
differentiate with respect to. Default: ``0``.
- **`create_graph`** – If ``True`` (default), the gradient is differentiable.
- **`realize`** – If ``True`` and *create_graph* is ``False``, eagerly
materialise outputs before returning.

**Returns**

A callable with the same signature as *fun* that returns
``(value, gradient)`` where *value* is the scalar output of *fun*
and *gradient* is its gradient.


---
## `vjp`

```python
def vjp(fn: 'Callable[..., Any]', *primals: 'Any', has_aux: 'bool' = False, create_graph: 'bool' = True) -> 'tuple[Any, Callable[..., tuple[Any, ...]]] | tuple[Any, Callable[..., tuple[Any, ...]], Any]':
```
Compute the Vector-Jacobian Product (VJP) of *fn* at *primals*.

Evaluates *fn* and returns a pullback function that multiplies a
cotangent vector by the Jacobian. This is the fundamental building
block for reverse-mode automatic differentiation.

**Parameters**

- **`fn`** – Differentiable function to differentiate.
- **`*primals`** – Input values at which to evaluate *fn* and the VJP.
- **`has_aux`** – If ``True``, *fn* must return ``(output, aux)``.
The auxiliary data *aux* is returned as a third element and
excluded from differentiation.
- **`create_graph`** – If ``True`` (default), the pullback is
differentiable, enabling higher-order AD.

**Returns**

- ``(output, pullback)`` when *has_aux* is ``False``.
- ``(output, pullback, aux)`` when *has_aux* is ``True``.

The returned *pullback* is a function that takes a cotangent
vector (with the same structure as *output*) and returns
a tuple of input cotangents.


---
## `jvp`

```python
def jvp(fn: 'Callable[..., Any]', primals: 'tuple[Any, ...]', tangents: 'tuple[Any, ...]', *, has_aux: 'bool' = False, create_graph: 'bool' = True) -> 'tuple[Any, Any] | tuple[Any, Any, Any]':
```
Compute the Jacobian-Vector Product (JVP) of *fn* at *primals*.

Pushes *tangents* through the computation graph via forward-mode AD.
Analogous to JAX's ``jax.jvp``.

**Parameters**

- **`fn`** – Differentiable function to differentiate.
- **`primals`** – Input values at which to evaluate *fn*. Must be a tuple.
- **`tangents`** – Tangent vectors aligned with *primals*. Must be a tuple
of the same length and structure.
- **`has_aux`** – If ``True``, *fn* must return ``(output, aux)``. The
auxiliary data is excluded from differentiation and returned
as the third element.
- **`create_graph`** – If ``True`` (default), the output tangents are
differentiable, enabling higher-order forward/reverse mixes.

**Returns**

- ``(output, tangent_out)`` when *has_aux* is ``False``.
- ``(output, tangent_out, aux)`` when *has_aux* is ``True``.


---
## `jacrev`

```python
def jacrev(fn: 'Callable[..., Any]', argnums: 'int | tuple[int, ...] | list[int] | None' = None, has_aux: 'bool' = False) -> 'Callable[..., Any]':
```
Compute the Jacobian of *fn* using reverse-mode autodiff.

Internally uses ``vmap`` over VJP cotangent directions, following the
same pattern as JAX. Composes naturally with other transforms.

**Parameters**

- **`fn`** – Differentiable function to differentiate.
- **`argnums`** – Index or list of indices of arguments to differentiate
with respect to. ``None`` differentiates all tensor arguments.
- **`has_aux`** – If ``True``, *fn* must return ``(output, aux)`` where
*aux* is not differentiated.

**Returns**

A callable that returns the Jacobian (or a tuple of Jacobians
when *argnums* selects multiple arguments). Shape of each Jacobian
is ``(*out_shape, *in_shape)``.


---
## `jacfwd`

```python
def jacfwd(fn: 'Callable[..., Any]', argnums: 'int | tuple[int, ...] | list[int] | None' = None, has_aux: 'bool' = False) -> 'Callable[..., Any]':
```
Compute the Jacobian of *fn* using forward-mode autodiff.

Internally uses ``vmap`` over JVP tangent directions, following the
same pattern as JAX. More efficient than :func:`jacrev` when the number
of input elements is smaller than the number of output elements.

**Parameters**

- **`fn`** – Differentiable function to differentiate.
- **`argnums`** – Index or list of indices of arguments to differentiate
with respect to. ``None`` differentiates all tensor arguments.
- **`has_aux`** – If ``True``, *fn* must return ``(output, aux)`` where
*aux* is not differentiated.

**Returns**

A callable that returns the Jacobian (or a tuple of Jacobians
when *argnums* selects multiple arguments). Shape of each Jacobian
is ``(*out_shape, *in_shape)``.


---
