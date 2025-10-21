# jacrev

## Signature

```python
nabla.jacrev(func: collections.abc.Callable[..., typing.Any], argnums: int | tuple[int, ...] | list[int] | None = None, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.jacrev`

Compute the Jacobian of a function using reverse-mode autodiff.

Args:
    func: Function to differentiate (should take positional arguments)
    argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether `func` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether `func` is promised to be
        holomorphic. Default False. Currently ignored.
    allow_int: Optional, bool. Whether to allow differentiating with
        respect to integer valued inputs. Currently ignored.

Returns:
    A function with the same arguments as `func`, that evaluates the Jacobian of
    `func` using reverse-mode automatic differentiation. If `has_aux` is True
    then a pair of (jacobian, auxiliary_data) is returned.

Note:
    This follows JAX's jacrev API:
    - Only accepts positional arguments
    - For functions requiring keyword arguments, use functools.partial or lambda
    - Returns the Jacobian as a pytree structure matching the input structure

Examples:
    Basic Jacobian computation:

    ```python
    import nabla as nb

    def f(x):
        return x ** 2

    x = nb.tensor([1.0, 2.0, 3.0])
    jac_fn = nb.jacrev(f)
    jacobian = jac_fn(x)
    # jacobian will be a diagonal matrix with [2.0, 4.0, 6.0] on the diagonal
    ```

    Vector-valued function:

    ```python
    import nabla as nb

    def f(x):
        return nb.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])

    x = nb.tensor([3.0, 4.0])
    jacobian = nb.jacrev(f)(x)
    # jacobian shape: (3, 2) - 3 outputs, 2 inputs
    ```

    Multiple arguments with argnums:

    ```python
    import nabla as nb

    def f(x, y):
        return x * y + x ** 2

    x = nb.tensor([1.0, 2.0])
    y = nb.tensor([3.0, 4.0])
    
    # Differentiate w.r.t. first argument
    jac_x = nb.jacrev(f, argnums=0)(x, y)
    
    # Differentiate w.r.t. both arguments
    jac_both = nb.jacrev(f, argnums=(0, 1))(x, y)
    ```

    With auxiliary data:

    ```python
    import nabla as nb

    def f_with_aux(x):
        y = x ** 2
        aux = {"loss": nb.sum(y)}
        return y, aux

    x = nb.tensor([1.0, 2.0, 3.0])
    jacobian, aux = nb.jacrev(f_with_aux, has_aux=True)(x)
    ```

