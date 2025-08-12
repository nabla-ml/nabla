# jacfwd

## Signature

```python
nabla.jacfwd(func: collections.abc.Callable[..., typing.Any], argnums: int | tuple[int, ...] | list[int] | None = None, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> collections.abc.Callable[..., typing.Any]
```

## Description

Prototype implementation of jacfwd using forward-mode autodiff.

This computes the Jacobian using the pattern:
vmap(jvp(func, primals, tangents), in_axes=(primal_axes, tangent_axes))

where primal_axes are None (broadcast) and tangent_axes are 0 (vectorize).


## Parameters

func: Function to differentiate
argnums: Which arguments to differentiate with respect to
has_aux: Whether function returns auxiliary data
holomorphic: Ignored (for JAX compatibility)
allow_int: Ignored (for JAX compatibility)


## Returns

Function that computes the Jacobian using forward-mode autodiff

