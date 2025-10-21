# negate

## Signature

```python
nabla.negate(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

Computes the element-wise numerical negative of an tensor.

This function returns a new tensor with each element being the negation
of the corresponding element in the input tensor. It also provides the
implementation for the unary `-` operator on Nabla tensors.

Parameters
----------
arg : Tensor
    The input tensor.

Returns
-------
Tensor
    An tensor containing the negated elements.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([1, -2, 3.5])
>>> nb.negate(x)
Tensor([-1.,  2., -3.5], dtype=float32)

Using the `-` operator:
>>> -x
Tensor([-1.,  2., -3.5], dtype=float32)

