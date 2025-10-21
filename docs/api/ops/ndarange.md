# ndarange

## Signature

```python
nabla.ndarange(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

Creates an tensor of a given shape with sequential values.

The tensor is filled with values from 0 to N-1, where N is the total
number of elements (the product of the shape dimensions).

Parameters
----------
shape : Shape
    The shape of the output tensor.
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
    An tensor of the specified shape containing values from 0 to N-1.

Examples
--------

.. code-block:: python

    >>> import nabla as nb
    >>> nb.ndarange((2, 3), dtype=nb.DType.int32)
    Tensor([[0, 1, 2],
    [3, 4, 5]], dtype=int32)

