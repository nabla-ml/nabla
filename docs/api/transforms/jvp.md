# jvp

## Signature

```python
nabla.jvp(func: collections.abc.Callable[..., typing.Any], primals, tangents, has_aux: bool = False) -> tuple[typing.Any, typing.Any] | tuple[typing.Any, typing.Any, typing.Any]
```

**Source**: `nabla.transforms.jvp`

## Description

Compute Jacobian-vector product (forward-mode autodiff).

Args:
    func: Function to differentiate (should take positional arguments)
    primals: Positional arguments to the function (can be arbitrary pytrees)
    tangents: Tangent vectors for directional derivatives (matching structure of primals)
    has_aux: Optional, bool. Indicates whether func returns a pair where the first element
        is considered the output of the mathematical function to be differentiated and the
        second element is auxiliary data. Default False.

## Returns

If has_aux is False, returns a (outputs, output_tangents) pair.
    If has_aux is True, returns a (outputs, output_tangents, aux) tuple where aux is the
    auxiliary data returned by func.

Note:
    This follows JAX's jvp API:
    - Only accepts positional arguments
    - For functions requiring keyword arguments, use functools.partial or lambda
