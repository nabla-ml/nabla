# div

## Signature

```python
nabla.div(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

**Source**: `nabla.ops.binary`

Divides two tensors element-wise.

This function performs element-wise (true) division on two tensors. It
supports broadcasting, allowing tensors of different shapes to be combined
as long as their shapes are compatible. This function also provides the
implementation of the `/` operator for Nabla tensors.

Parameters
----------
x : Tensor | float | int
    The first input tensor or scalar (the dividend).
y : Tensor | float | int
    The second input tensor or scalar (the divisor). Must be broadcastable
    to the same shape as `x`.

Returns
-------
Tensor
    An tensor containing the result of the element-wise division. The
    result is typically a floating-point tensor.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([10, 20, 30])
>>> y = nb.tensor([2, 5, 10])
>>> nb.div(x, y)
Tensor([5., 4., 3.], dtype=float32)

>>> x / y
Tensor([5., 4., 3.], dtype=float32)

