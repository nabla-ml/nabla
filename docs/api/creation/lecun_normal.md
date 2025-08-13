# lecun_normal

## Signature

```python
nabla.lecun_normal(shape: 'Shape', dtype: 'DType', device: 'Device', seed: 'int', batch_dims: 'Shape', traced: 'bool') -> 'Array'
```

## Description

Fills an array with values according to the LeCun normal initializer.

This method is often used for layers with SELU activations. It samples from
a normal distribution N(0, std^2) where std = sqrt(1 / fan_in).

## Parameters

- **`shape`** (`Shape`): The shape of the output array. Must be at least 2D.

- **`dtype`** (`DType, optional`): The desired data type for the array. Defaults to DType.float32.

- **`device`** (`Device, optional`): The device to place the array on. Defaults to the CPU.

- **`seed`** (`int, optional`): The seed for the random number generator. Defaults to 0.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

## Returns

- `Array`: An array initialized with the LeCun normal distribution.
