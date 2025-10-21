# pow

## Signature

```python
nabla.pow(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

**Source**: `nabla.ops.binary`

Computes `x` raised to the power of `y` element-wise.

This function calculates `x ** y` for each element in the input tensors.
It supports broadcasting and provides the implementation of the `**`
operator for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The base tensor or scalar.
y : Tensor | float | int
    The exponent tensor or scalar. Must be broadcastable to the same shape
    as `x`.

Returns
-------
Tensor
    An tensor containing the result of the element-wise power operation.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([2, 3, 2])
>>> nb.pow(x, y)
Tensor([1, 8, 9], dtype=int32)

>>> x ** y
Tensor([1, 8, 9], dtype=int32)

