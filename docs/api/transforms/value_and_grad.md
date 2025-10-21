# value_and_grad

## Signature

```python
nabla.value_and_grad(fun: collections.abc.Callable | None = None, argnums: int | collections.abc.Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False, reduce_axes: collections.abc.Sequence = ()) -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.grad`

## Description

Creates a function that evaluates both the value and gradient of fun.

This function uses VJP (Vector-Jacobian Product) directly with a cotangent
of ones_like(output) to compute gradients for scalar-valued functions.
This is simpler and more efficient than using jacrev/jacfwd for scalar outputs.

Parameters:
    fun: Function to be differentiated. Should return a scalar.
    argnums: Which positional argument(s) to differentiate with respect to (default 0).
    has_aux: Whether fun returns (output, aux) pair (default False).
    holomorphic: Whether fun is holomorphic - currently ignored (default False).
    allow_int: Whether to allow integer inputs - currently ignored (default False).
    reduce_axes: Axes to reduce over - currently ignored (default ()).

Returns:
    A function that computes both the value and gradient of fun.

Examples:
    Basic usage as a function call::

        value_and_grad_fn = value_and_grad(my_loss)
        value, grads = value_and_grad_fn(x)

    Usage as a decorator::

        @value_and_grad
        def my_loss(x):
            return x**2

        value, grads = my_loss(3.0)
