# vjp

## Signature

```python
nabla.vjp(func: collections.abc.Callable[..., typing.Any], *primals, has_aux: bool = False) -> tuple[typing.Any, collections.abc.Callable] | tuple[typing.Any, collections.abc.Callable, typing.Any]
```

**Source**: `nabla.transforms.vjp`

Compute vector-Jacobian product (reverse-mode autodiff).

Parameters
----------
func : Callable
    Function to differentiate (should take positional arguments)
*primals : tuple
    Positional arguments to the function (can be arbitrary pytrees)
has_aux : bool, optional
    Indicates whether `func` returns a pair where the
    first element is considered the output of the mathematical function to be
    differentiated and the second element is auxiliary data. Default False.

Returns
-------
tuple
    If has_aux is False: Tuple of (outputs, vjp_function) where vjp_function computes gradients.
    If has_aux is True: Tuple of (outputs, vjp_function, aux) where aux is the auxiliary data.
    
    The vjp_function always returns gradients as a tuple (matching JAX behavior):
    - Single argument: vjp_fn(cotangent) -> (gradient,)
    - Multiple arguments: vjp_fn(cotangent) -> (grad1, grad2, ...)

Notes
-----
This follows JAX's vjp API exactly:
- Only accepts positional arguments
- Always returns gradients as tuple
- For functions requiring keyword arguments, use functools.partial or lambda

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> def f(x):
    ...     return x ** 2
    >>> primals = nb.tensor(3.0)
    >>> y, vjp_fn = nb.vjp(f, primals)
    >>> cotangent = nb.tensor(1.0)
    >>> (grad_x,) = vjp_fn(cotangent)

Multiple inputs:


.. code-block:: python

    >>> def f(x, y):
    ...     return x * y + x ** 2
    >>> x = nb.tensor(3.0)
    >>> y = nb.tensor(4.0)
    >>> output, vjp_fn = nb.vjp(f, x, y)
    >>> cotangent = nb.tensor(1.0)
    >>> grad_x, grad_y = vjp_fn(cotangent)

