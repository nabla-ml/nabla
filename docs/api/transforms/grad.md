# grad

## Signature

```python
nabla.grad(fun: collections.abc.Callable | None = None, argnums: int | collections.abc.Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False, reduce_axes: collections.abc.Sequence = (), mode: str = 'reverse') -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.grad`

Creates a function that evaluates the gradient of fun.

This is implemented as a special case of value_and_grad that only returns
the gradient part. Uses VJP directly for efficiency with scalar outputs.

Parameters
----------
fun : Callable or None
    Function to be differentiated. Should return a scalar.
argnums : int or Sequence[int], optional
    Which positional argument(s) to differentiate with respect to (default 0).
has_aux : bool, optional
    Whether fun returns (output, aux) pair (default False).
holomorphic : bool, optional
    Whether fun is holomorphic - currently ignored (default False).
allow_int : bool, optional
    Whether to allow integer inputs - currently ignored (default False).
reduce_axes : Sequence, optional
    Axes to reduce over - currently ignored (default ()).
mode : str, optional
    Kept for API compatibility but ignored (always uses reverse-mode VJP).

Returns
-------
Callable
    A function that computes the gradient of fun.

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> def my_loss(x):
    ...     return x**2
    >>> grad_fn = nb.grad(my_loss)
    >>> grads = grad_fn(nb.tensor(3.0))

Usage as a decorator:


.. code-block:: python

    >>> @nb.grad
    ... def my_loss(x):
    ...     return x**2
    >>> grads = my_loss(nb.tensor(3.0))  # Returns gradient, not function value

