# ones

## Signature

```python
nabla.ones(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

Creates an tensor of a given shape filled with ones.

Parameters
----------
shape : Shape
    The shape of the new tensor, e.g., `(2, 3)` or `(5,)`.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor of the specified shape and dtype, filled with ones.

Examples
--------
>>> import nabla as nb
>>> # Create a vector of ones
>>> nb.ones((4,), dtype=nb.DType.float32)
Tensor([1., 1., 1., 1.], dtype=float32)

