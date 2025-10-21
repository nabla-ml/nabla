# add

## Signature

```python
nabla.add(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

**Source**: `nabla.ops.binary`

Adds two tensors element-wise.

This function performs element-wise addition on two tensors. It supports
broadcasting, allowing tensors of different shapes to be combined as long
as their shapes are compatible. This function also provides the
implementation of the `+` operator for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The first input tensor or scalar.
y : Tensor | float | int
    The second input tensor or scalar. Must be broadcastable to the same
    shape as `x`.

Returns
-------
Tensor
    An tensor containing the result of the element-wise addition.

Examples
--------
Calling `add` explicitly:

>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([4, 5, 6])
>>> nb.add(x, y)
Tensor([5, 7, 9], dtype=int32)

Calling `add` via the `+` operator:

>>> x + y
Tensor([5, 7, 9], dtype=int32)

Broadcasting a scalar:

>>> x + 10
Tensor([11, 12, 13], dtype=int32)

