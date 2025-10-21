# jacfwd

## Signature

```python
nabla.jacfwd(func: collections.abc.Callable[..., typing.Any], argnums: int | tuple[int, ...] | list[int] | None = None, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.jacfwd`

Prototype implementation of jacfwd using forward-mode autodiff.

This computes the Jacobian using the pattern:
vmap(jvp(func, primals, tangents), in_axes=(primal_axes, tangent_axes))

where primal_axes are None (broadcast) and tangent_axes are 0 (vectorize).

Args:
    func: Function to differentiate
    argnums: Which arguments to differentiate with respect to
    has_aux: Whether function returns auxiliary data
    holomorphic: Ignored (for JAX compatibility)
    allow_int: Ignored (for JAX compatibility)

Returns:
    Function that computes the Jacobian using forward-mode autodiff

Examples:
    Basic Jacobian computation:

    ```python
    import nabla as nb

    def f(x):
        return x ** 2

    x = nb.tensor([1.0, 2.0, 3.0])
    jac_fn = nb.jacfwd(f)
    jacobian = jac_fn(x)
    # jacobian will be a diagonal matrix with [2.0, 4.0, 6.0] on the diagonal
    ```

    Vector-valued function:

    ```python
    import nabla as nb

    def f(x):
        return nb.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])

    x = nb.tensor([3.0, 4.0])
    jacobian = nb.jacfwd(f)(x)
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
    jac_x = nb.jacfwd(f, argnums=0)(x, y)
    
    # Differentiate w.r.t. both arguments
    jac_both = nb.jacfwd(f, argnums=(0, 1))(x, y)
    ```

