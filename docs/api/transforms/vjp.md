# vjp

## Signature

```python
nabla.vjp(func: collections.abc.Callable[..., typing.Any], *primals, has_aux: bool = False) -> tuple[typing.Any, collections.abc.Callable] | tuple[typing.Any, collections.abc.Callable, typing.Any]
```

**Source**: `nabla.transforms.vjp`

Compute vector-Jacobian product (reverse-mode autodiff).

Args:
    func: Function to differentiate (should take positional arguments)
    *primals: Positional arguments to the function (can be arbitrary pytrees)
    has_aux: Optional, bool. Indicates whether `func` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.

Returns:
    If has_aux is False:
        Tuple of (outputs, vjp_function) where vjp_function computes gradients.
    If has_aux is True:
        Tuple of (outputs, vjp_function, aux) where aux is the auxiliary data.

    The vjp_function always returns gradients as a tuple (matching JAX behavior):
    - Single argument: vjp_fn(cotangent) -> (gradient,)
    - Multiple arguments: vjp_fn(cotangent) -> (grad1, grad2, ...)

    Note:
    This follows JAX's vjp API exactly:
    - Only accepts positional arguments
    - Always returns gradients as tuple
    - For functions requiring keyword arguments, use functools.partial or lambda

Examples:
    Basic reverse-mode differentiation:

    ```python
    import nabla as nb

    def f(x):
        return x ** 2

    # Forward pass
    primals = nb.tensor(3.0)
    y, vjp_fn = nb.vjp(f, primals)
    
    # Backward pass with cotangent
    cotangent = nb.tensor(1.0)
    (grad_x,) = vjp_fn(cotangent)
    # grad_x = 6.0 (derivative of x^2 at x=3 is 2*3=6)
    ```

    Multiple inputs:

    ```python
    import nabla as nb

    def f(x, y):
        return x * y + x ** 2

    x = nb.tensor(3.0)
    y = nb.tensor(4.0)
    output, vjp_fn = nb.vjp(f, x, y)
    
    cotangent = nb.tensor(1.0)
    grad_x, grad_y = vjp_fn(cotangent)
    ```

    With auxiliary data:

    ```python
    import nabla as nb

    def f_with_aux(x):
        y = x ** 2
        aux = {"intermediate": x * 2}
        return y, aux

    x = nb.tensor(3.0)
    y, vjp_fn, aux = nb.vjp(f_with_aux, x, has_aux=True)
    
    cotangent = nb.tensor(1.0)
    (grad_x,) = vjp_fn(cotangent)
    ```

