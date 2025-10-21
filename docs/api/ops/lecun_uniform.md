# lecun_uniform

## Signature

```python
nabla.lecun_uniform(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

Fills an tensor with values according to the LeCun uniform initializer.

This method is often used for layers with SELU activations. It samples from
a uniform distribution U(-a, a) where a = sqrt(3 / fan_in).

Parameters
----------
shape : Shape
    The shape of the output tensor. Must be at least 2D.
dtype : DType, optional
    The desired data type for the tensor. Defaults to DType.float32.
device : Device, optional
    The device to place the tensor on. Defaults to the CPU.
seed : int, optional
    The seed for the random number generator. Defaults to 0.
batch_dims : Shape, optional
    Specifies leading dimensions to be treated as batch dimensions.
    Defaults to an empty tuple.
traced : bool, optional
    Whether the operation should be traced in the graph. Defaults to False.

Returns
-------
Tensor
    An tensor initialized with the LeCun uniform distribution.

