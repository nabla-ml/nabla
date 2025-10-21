# Automatic Differentiation

## `grad`

```python
def grad(fun: collections.abc.Callable | None = None, argnums: int | collections.abc.Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False, reduce_axes: collections.abc.Sequence = (), mode: str = 'reverse') -> collections.abc.Callable[..., typing.Any]:
```
Creates a function that evaluates the gradient of fun.

This is implemented as a special case of value_and_grad that only returns
the gradient part. Uses VJP directly for efficiency with scalar outputs.

**Parameters**

- **`fun`** : `Callable or None` – Function to be differentiated. Should return a scalar.
- **`argnums`** : `int or Sequence[int]`, optional – Which positional argument(s) to differentiate with respect to (default 0).
- **`has_aux`** : `bool`, optional – Whether fun returns (output, aux) pair (default False).
- **`holomorphic`** : `bool`, optional – Whether fun is holomorphic - currently ignored (default False).
- **`allow_int`** : `bool`, optional – Whether to allow integer inputs - currently ignored (default False).
- **`reduce_axes`** : `Sequence`, optional – Axes to reduce over - currently ignored (default ()).
- **`mode`** : `str`, optional – Kept for API compatibility but ignored (always uses reverse-mode VJP).

**Returns**

`Callable` – A function that computes the gradient of fun.

**Examples**

...     return x**2
```python
>>> import nabla as nb
>>> def my_loss(x):
```


Usage as a decorator:

```python
>>> grad_fn = nb.grad(my_loss)
>>> grads = grad_fn(nb.tensor(3.0))
```

... def my_loss(x):
...     return x**2
```python
>>> @nb.grad
```

```python
>>> grads = my_loss(nb.tensor(3.0))  # Returns gradient, not function value
```


---
## `value_and_grad`

```python
def value_and_grad(fun: collections.abc.Callable | None = None, argnums: int | collections.abc.Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False, reduce_axes: collections.abc.Sequence = ()) -> collections.abc.Callable[..., typing.Any]:
```
Creates a function that evaluates both the value and gradient of fun.

This function uses VJP (Vector-Jacobian Product) directly with a cotangent
of ones_like(output) to compute gradients for scalar-valued functions.
This is simpler and more efficient than using jacrev/jacfwd for scalar outputs.

**Parameters**

- **`fun`** : `Callable or None` – Function to be differentiated. Should return a scalar.
- **`argnums`** : `int or Sequence[int]`, optional – Which positional argument(s) to differentiate with respect to (default 0).
- **`has_aux`** : `bool`, optional – Whether fun returns (output, aux) pair (default False).
- **`holomorphic`** : `bool`, optional – Whether fun is holomorphic - currently ignored (default False).
- **`allow_int`** : `bool`, optional – Whether to allow integer inputs - currently ignored (default False).
- **`reduce_axes`** : `Sequence`, optional – Axes to reduce over - currently ignored (default ()).

**Returns**

`Callable` – A function that computes both the value and gradient of fun.

**Examples**

...     return x**2
```python
>>> import nabla as nb
>>> def my_loss(x):
```


Usage as a decorator:

```python
>>> value_and_grad_fn = nb.value_and_grad(my_loss)
>>> value, grads = value_and_grad_fn(nb.tensor(3.0))
```

... def my_loss(x):
...     return x**2
```python
>>> @nb.value_and_grad
```

```python
>>> value, grads = my_loss(nb.tensor(3.0))
```


---
## `vjp`

```python
def vjp(func: collections.abc.Callable[..., typing.Any], *primals, has_aux: bool = False) -> tuple[typing.Any, collections.abc.Callable] | tuple[typing.Any, collections.abc.Callable, typing.Any]:
```
Compute vector-Jacobian product (reverse-mode autodiff).

**Parameters**

- **`func`** : `Callable` – Function to differentiate (should take positional arguments)
- **`*primals`** : `tuple` – Positional arguments to the function (can be arbitrary pytrees)
- **`has_aux`** : `bool`, optional, default: `False` – Indicates whether `func` returns a pair where the
first element is considered the output of the mathematical function to be
differentiated and the second element is auxiliary data. Default False.

**Returns**

`tuple` – If has_aux is False: Tuple of (outputs, vjp_function) where vjp_function computes gradients.
If has_aux is True: Tuple of (outputs, vjp_function, aux) where aux is the auxiliary data.

The vjp_function always returns gradients as a tuple (matching JAX behavior):
- Single argument: vjp_fn(cotangent) -> (gradient,)
- Multiple arguments: vjp_fn(cotangent) -> (grad1, grad2, ...)

**Examples**

...     return x ** 2
```python
>>> import nabla as nb
>>> def f(x):
```


Multiple inputs:

```python
>>> primals = nb.tensor(3.0)
>>> y, vjp_fn = nb.vjp(f, primals)
>>> cotangent = nb.tensor(1.0)
>>> (grad_x,) = vjp_fn(cotangent)
```

...     return x * y + x ** 2
```python
>>> def f(x, y):
```

```python
>>> x = nb.tensor(3.0)
>>> y = nb.tensor(4.0)
>>> output, vjp_fn = nb.vjp(f, x, y)
>>> cotangent = nb.tensor(1.0)
>>> grad_x, grad_y = vjp_fn(cotangent)
```


---
## `jvp`

```python
def jvp(func: collections.abc.Callable[..., typing.Any], primals, tangents, has_aux: bool = False) -> tuple[typing.Any, typing.Any] | tuple[typing.Any, typing.Any, typing.Any]:
```
Compute Jacobian-vector product (forward-mode autodiff).

**Parameters**

- **`func`** : `Callable` – Function to differentiate (should take positional arguments)
- **`primals`** : `tuple or pytree` – Positional arguments to the function (can be arbitrary pytrees)
- **`tangents`** : `tuple or pytree` – Tangent vectors for directional derivatives (matching structure of primals)
- **`has_aux`** : `bool`, optional, default: `False` – Indicates whether func returns a pair where the first element
is considered the output of the mathematical function to be differentiated and the
second element is auxiliary data. Default False.

**Returns**

`tuple` – If has_aux is False, returns a (outputs, output_tangents) pair.
If has_aux is True, returns a (outputs, output_tangents, aux) tuple where aux is the
auxiliary data returned by func.

**Examples**

...     return x ** 2
```python
>>> import nabla as nb
>>> def f(x):
```


Multiple inputs:

```python
>>> primals = (nb.tensor(3.0),)
>>> tangents = (nb.tensor(1.0),)
>>> y, y_dot = nb.jvp(f, primals, tangents)
```

...     return x * y + x ** 2
```python
>>> def f(x, y):
```

```python
>>> primals = (nb.tensor(3.0), nb.tensor(4.0))
>>> tangents = (nb.tensor(1.0), nb.tensor(0.0))
>>> output, tangent_out = nb.jvp(f, primals, tangents)
```


---
## `jacfwd`

```python
def jacfwd(func: collections.abc.Callable[..., typing.Any], argnums: int | tuple[int, ...] | list[int] | None = None, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> collections.abc.Callable[..., typing.Any]:
```
Prototype implementation of jacfwd using forward-mode autodiff.

This computes the Jacobian using the pattern:
vmap(jvp(func, primals, tangents), in_axes=(primal_axes, tangent_axes))

where primal_axes are None (broadcast) and tangent_axes are 0 (vectorize).

**Parameters**

- **`func`** : `Callable` – Function to differentiate
- **`argnums`** : `int or tuple of int or list of int or None`, optional – Which arguments to differentiate with respect to
- **`has_aux`** : `bool`, optional – Whether function returns auxiliary data
- **`holomorphic`** : `bool`, optional – Ignored (for JAX compatibility)
- **`allow_int`** : `bool`, optional – Ignored (for JAX compatibility)

**Returns**

`Callable` – Function that computes the Jacobian using forward-mode autodiff

**Examples**

...     return x ** 2
```python
>>> import nabla as nb
>>> def f(x):
```


Vector-valued function:

```python
>>> x = nb.tensor([1.0, 2.0, 3.0])
>>> jac_fn = nb.jacfwd(f)
>>> jacobian = jac_fn(x)
```

...     return nb.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])
```python
>>> def f(x):
```


Multiple arguments with argnums:

```python
>>> x = nb.tensor([3.0, 4.0])
>>> jacobian = nb.jacfwd(f)(x)
```

...     return x * y + x ** 2
```python
>>> def f(x, y):
```

```python
>>> x = nb.tensor([1.0, 2.0])
>>> y = nb.tensor([3.0, 4.0])
>>> jac_x = nb.jacfwd(f, argnums=0)(x, y)
>>> jac_both = nb.jacfwd(f, argnums=(0, 1))(x, y)
```


---
## `jacrev`

```python
def jacrev(func: collections.abc.Callable[..., typing.Any], argnums: int | tuple[int, ...] | list[int] | None = None, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> collections.abc.Callable[..., typing.Any]:
```
Compute the Jacobian of a function using reverse-mode autodiff.

**Parameters**

- **`func`** : `Callable` – Function to differentiate (should take positional arguments)
- **`argnums`** : `int or tuple of int or list of int or None`, optional – Specifies which positional argument(s) to differentiate with respect to (default 0).
- **`has_aux`** : `bool`, optional, default: `False` – Indicates whether `func` returns a pair where the
first element is considered the output of the mathematical function to be
differentiated and the second element is auxiliary data. Default False.
- **`holomorphic`** : `bool`, optional, default: `False` – Indicates whether `func` is promised to be holomorphic. Default False. Currently ignored.
- **`allow_int`** : `bool`, optional – Whether to allow differentiating with respect to integer valued inputs. Currently ignored.

**Returns**

`Callable` – A function with the same arguments as `func`, that evaluates the Jacobian of
`func` using reverse-mode automatic differentiation. If `has_aux` is True
then a pair of (jacobian, auxiliary_data) is returned.

**Examples**

...     return x ** 2
```python
>>> import nabla as nb
>>> def f(x):
```


Vector-valued function:

```python
>>> x = nb.tensor([1.0, 2.0, 3.0])
>>> jac_fn = nb.jacrev(f)
>>> jacobian = jac_fn(x)
```

...     return nb.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])
```python
>>> def f(x):
```


Multiple arguments with argnums:

```python
>>> x = nb.tensor([3.0, 4.0])
>>> jacobian = nb.jacrev(f)(x)
```

...     return x * y + x ** 2
```python
>>> def f(x, y):
```

```python
>>> x = nb.tensor([1.0, 2.0])
>>> y = nb.tensor([3.0, 4.0])
>>> jac_x = nb.jacrev(f, argnums=0)(x, y)
>>> jac_both = nb.jacrev(f, argnums=(0, 1))(x, y)
```


---
## `backward`

```python
def backward(outputs: 'Any', cotangents: 'Any', retain_graph: 'bool' = False) -> 'None':
```
Accumulate gradients on traced leaf inputs for the given traced outputs.

**Parameters**

- **`outputs`** – Output tensors to backpropagate from
- **`cotangents`** – Cotangent vectors for outputs
- **`retain_graph`** – If False (default), frees the computation graph after backward pass


---
