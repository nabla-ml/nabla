# vmap

## Signature

```python
nabla.vmap(func=None, in_axes=0, out_axes=0) -> 'Callable[..., Any]'
```

## Description

Vectorize a function over specified input axes.
This can be used as a function call like `vmap(func, in_axes=0)` or as a decorator `@vmap`.


## Parameters

func: Function to vectorize
in_axes: Specification of axes to map over for inputs.
If an integer, all inputs are mapped over that axis.
If a tuple, should match the length of inputs with axis specifications.
out_axes: Specification of axes for outputs.
If an integer, all outputs are mapped over that axis.
If a tuple, should match the structure of outputs.


## Returns

Vectorized function that can handle batched inputs


## Examples

As a function call::

vmapped_func = vmap(my_func, in_axes=0, out_axes=0)

As a decorator::

@vmap
def my_func(x):
return x * 2

As a decorator with arguments::

@vmap(in_axes=1, out_axes=0)
def my_func(x):
return x * 2

## Notes

Supports both calling conventions:

* List-style: vmapped_fn([x, y, z])
* Unpacked-style: vmapped_fn(x, y, z)


## See Also

- {doc}`jit <core_jit>` - Just-in-time compilation
- {doc}`vjp <core_vjp>`, {doc}`jvp <core_jvp>` - Automatic differentiation

