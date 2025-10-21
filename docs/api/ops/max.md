# max

## Signature

```python
nabla.max(arg: 'Tensor', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.reduce`

Finds the maximum value of tensor elements over given axes.

This function reduces an tensor by finding the maximum element along the
specified axes. If no axes are provided, the maximum of all elements in the
tensor is returned.

Parameters
----------
arg : Tensor
    The input tensor.
axes : int | list[int] | tuple[int, ...] | None, optional
    The axis or axes along which to find the maximum. If None (the
    default), the maximum is found over all axes, resulting in a scalar
    tensor.
keep_dims : bool, optional
    If True, the axes which are reduced are left in the result as
    dimensions with size one. This allows the result to broadcast
    correctly against the original tensor. Defaults to False.

Returns
-------
Tensor
    An tensor containing the maximum values.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([[1, 5, 2], [4, 3, 6]])

Find the maximum of all elements:
>>> nb.max(x)
Tensor([6], dtype=int32)

Find the maximum along an axis:
>>> nb.max(x, axes=1)
Tensor([5, 6], dtype=int32)

Find the maximum along an axis and keep dimensions:
>>> nb.max(x, axes=0, keep_dims=True)
Tensor([[4, 5, 6]], dtype=int32)

