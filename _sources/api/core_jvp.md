# jvp

## Signature

```python
nabla.jvp(func: 'Callable[..., Any]', primals, tangents) -> 'tuple[Any, Any]'
```

## Description

Compute Jacobian-vector product (forward-mode autodiff).


## Parameters

func: Function to differentiate (should take positional arguments)
primals: Positional arguments to the function (can be arbitrary pytrees)
tangents: Tangent vectors for directional derivatives (matching structure of primals)


## Returns

Tuple of (outputs, output_tangents) where output_tangents are the JVP results


## Notes

This follows JAX's jvp API:
- Only accepts positional arguments
- For functions requiring keyword arguments, use functools.partial or lambda

## See Also

- {doc}`vjp <core_vjp>` - Vector-Jacobian product

