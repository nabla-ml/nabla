# jvp

## Signature

```python
nabla.jvp(func: collections.abc.Callable[..., typing.Any], primals, tangents, has_aux: bool = False) -> tuple[typing.Any, typing.Any] | tuple[typing.Any, typing.Any, typing.Any]
```

**Source**: `nabla.transforms.jvp`

Compute Jacobian-vector product (forward-mode autodiff).

Parameters
----------
func : Callable
    Function to differentiate (should take positional arguments)
primals : tuple or pytree
    Positional arguments to the function (can be arbitrary pytrees)
tangents : tuple or pytree
    Tangent vectors for directional derivatives (matching structure of primals)
has_aux : bool, optional
    Indicates whether func returns a pair where the first element
    is considered the output of the mathematical function to be differentiated and the
    second element is auxiliary data. Default False.

Returns
-------
tuple
    If has_aux is False, returns a (outputs, output_tangents) pair.
    If has_aux is True, returns a (outputs, output_tangents, aux) tuple where aux is the
    auxiliary data returned by func.

Notes
-----
This follows JAX's jvp API:
- Only accepts positional arguments
- For functions requiring keyword arguments, use functools.partial or lambda

Examples
--------
>>> import nabla as nb
>>> def f(x):
...     return x ** 2
>>> primals = (nb.tensor(3.0),)
>>> tangents = (nb.tensor(1.0),)
>>> y, y_dot = nb.jvp(f, primals, tangents)

Multiple inputs:

>>> def f(x, y):
...     return x * y + x ** 2
>>> primals = (nb.tensor(3.0), nb.tensor(4.0))
>>> tangents = (nb.tensor(1.0), nb.tensor(0.0))
>>> output, tangent_out = nb.jvp(f, primals, tangents)

