# decr_batch_dim_ctr

## Signature

```python
nabla.decr_batch_dim_ctr(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

Moves the last `batch_dim` to be the leading axis of `shape`. (Internal use)

This is an internal-use function primarily for developing function
transformations like `vmap`. It re-interprets the last batch dimension
as the new first dimension of the tensor's logical shape.

Parameters
----------
arg : Tensor
    The input tensor.

Returns
-------
Tensor
    A new tensor with one fewer batch dimension.

