# jacrev

## Signature

```python
nabla.jacrev(func: collections.abc.Callable[..., typing.Any], argnums: int | tuple[int, ...] | list[int] | None = None, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> collections.abc.Callable[..., typing.Any]
```

## Description

Compute the Jacobian of a function using reverse-mode autodiff.


## Parameters

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


## Returns

A function with the same arguments as `func`, that evaluates the Jacobian of
`func` using reverse-mode automatic differentiation. If `has_aux` is True
then a pair of (jacobian, auxiliary_data) is returned.


## Notes

This follows JAX's jacrev API:
- Only accepts positional arguments
- For functions requiring keyword arguments, use functools.partial or lambda
- Returns the Jacobian as a pytree structure matching the input structure

