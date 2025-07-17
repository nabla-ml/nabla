# vjp

## Signature

```python
nabla.vjp(func: collections.abc.Callable[..., typing.Any], *primals, has_aux: bool = False) -> tuple[typing.Any, collections.abc.Callable] | tuple[typing.Any, collections.abc.Callable, typing.Any]
```

## Description

Compute vector-Jacobian product (reverse-mode autodiff).


## Parameters

func: Function to differentiate (should take positional arguments)
*primals: Positional arguments to the function (can be arbitrary pytrees)
has_aux: Optional, bool. Indicates whether `func` returns a pair where the
first element is considered the output of the mathematical function to be
differentiated and the second element is auxiliary data. Default False.


## Returns

If has_aux is False:
Tuple of (outputs, vjp_function) where vjp_function computes gradients.
If has_aux is True:
Tuple of (outputs, vjp_function, aux) where aux is the auxiliary data.

The vjp_function always returns gradients as a tuple (matching JAX behavior):
- Single argument: vjp_fn(cotangent) -> (gradient,)
- Multiple arguments: vjp_fn(cotangent) -> (grad1, grad2, ...)


## Examples

```python
import nabla as nb

# Vector-Jacobian product for reverse-mode AD
def f(x):
    return nb.sum(x ** 2)

x = nb.array([1.0, 2.0, 3.0])
output, vjp_fn = nb.vjp(f, x)
gradients = vjp_fn(nb.ones_like(output))
print(gradients)  # [2.0, 4.0, 6.0]
```

## Notes

This follows JAX's vjp API exactly:
- Only accepts positional arguments
- Always returns gradients as tuple
- For functions requiring keyword arguments, use functools.partial or lambda

## See Also

- {doc}`jvp <core_jvp>` - Jacobian-vector product
- {doc}`grad <core_grad>` - Automatic differentiation

