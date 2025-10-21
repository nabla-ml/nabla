# sin

## Signature

```python
nabla.sin(arg: nabla.core.tensor.Tensor, dtype: max._core.dtype.DType | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

Computes the element-wise sine of an tensor.

Parameters
----------
arg : Tensor
    The input tensor. Input is expected to be in radians.
dtype : DType | None, optional
    If provided, the output tensor will be cast to this data type.

Returns
-------
Tensor
    An tensor containing the sine of each element in the input.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([0, 1.5707963, 3.1415926])
>>> nb.sin(x)
Tensor([0.0000000e+00, 1.0000000e+00, -8.7422780e-08], dtype=float32)

