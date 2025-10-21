# jvp

## Signature

```python
nabla.jvp(func: collections.abc.Callable[..., typing.Any], primals, tangents, has_aux: bool = False) -> tuple[typing.Any, typing.Any] | tuple[typing.Any, typing.Any, typing.Any]
```

**Source**: `nabla.transforms.jvp`

Compute Jacobian-vector product (forward-mode autodiff).

Args:
    func: Function to differentiate (should take positional arguments)
    primals: Positional arguments to the function (can be arbitrary pytrees)
    tangents: Tangent vectors for directional derivatives (matching structure of primals)
    has_aux: Optional, bool. Indicates whether func returns a pair where the first element
        is considered the output of the mathematical function to be differentiated and the
        second element is auxiliary data. Default False.

Returns:
    If has_aux is False, returns a (outputs, output_tangents) pair.
    If has_aux is True, returns a (outputs, output_tangents, aux) tuple where aux is the
    auxiliary data returned by func.

Note:
    This follows JAX's jvp API:
    - Only accepts positional arguments
    - For functions requiring keyword arguments, use functools.partial or lambda

Examples:
    Basic forward-mode differentiation:

    ```python
    import nabla as nb

    def f(x):
        return x ** 2

    # Compute derivative at x=3.0 in direction v=1.0
    primals = (nb.tensor(3.0),)
    tangents = (nb.tensor(1.0),)
    y, y_dot = nb.jvp(f, primals, tangents)
    # y = 9.0, y_dot = 6.0 (derivative of x^2 at x=3 is 2*3=6)
    ```

    Multiple inputs:

    ```python
    import nabla as nb

    def f(x, y):
        return x * y + x ** 2

    primals = (nb.tensor(3.0), nb.tensor(4.0))
    tangents = (nb.tensor(1.0), nb.tensor(0.0))  # Differentiate w.r.t. x only
    output, tangent_out = nb.jvp(f, primals, tangents)
    ```

    With auxiliary data:

    ```python
    import nabla as nb

    def f_with_aux(x):
        y = x ** 2
        aux = {"intermediate": x * 2}
        return y, aux

    primals = (nb.tensor(3.0),)
    tangents = (nb.tensor(1.0),)
    y, y_dot, aux = nb.jvp(f_with_aux, primals, tangents, has_aux=True)
    ```

