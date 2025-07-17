# grad

## Signature

```python
nabla.grad(fun: collections.abc.Callable | None = None, argnums: int | collections.abc.Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False, reduce_axes: collections.abc.Sequence = (), mode: str = 'reverse') -> collections.abc.Callable[..., typing.Any]
```

## Description

Creates a function that evaluates the gradient of fun.

This is implemented as a special case of value_and_grad that only returns
the gradient part. Uses VJP directly for efficiency with scalar outputs.


## Parameters

fun: Function to be differentiated. Should return a scalar.
argnums: Which positional argument(s) to differentiate with respect to (default 0).
has_aux: Whether fun returns (output, aux) pair (default False).
holomorphic: Whether fun is holomorphic - currently ignored (default False).
allow_int: Whether to allow integer inputs - currently ignored (default False).
reduce_axes: Axes to reduce over - currently ignored (default ()).
mode: Kept for API compatibility but ignored (always uses reverse-mode VJP).


## Returns

A function that computes the gradient of fun.


## Examples

Basic usage as a function call::

grad_fn = grad(my_loss)
grads = grad_fn(x)

Usage as a decorator::

@grad
def my_loss(x):
return x**2

grads = my_loss(3.0)  # Returns gradient, not function value

