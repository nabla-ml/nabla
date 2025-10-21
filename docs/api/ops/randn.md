# randn

## Signature

```python
nabla.randn(shape: 'Shape', dtype: 'DType' = float32, mean: 'float' = 0.0, std: 'float' = 1.0, device: 'Device' = Device(type=cpu,id=0), seed: 'int' = 0, batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

## Description

Creates an tensor with normally distributed random values.

The values are drawn from a normal (Gaussian) distribution with the
specified mean and standard deviation.

## Parameters

- **`shape`** (`Shape`): The shape of the output tensor.

- **`dtype`** (`DType, optional`): The desired data type for the tensor. Defaults to DType.float32.

- **`mean`** (`float, optional`): The mean of the normal distribution. Defaults to 0.0.

- **`std`** (`float, optional`): The standard deviation of the normal distribution. Defaults to 1.0.

- **`device`** (`Device, optional`): The device to place the tensor on. Defaults to the CPU.

- **`seed`** (`int, optional`): The seed for the random number generator for reproducibility. Defaults to 0.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

## Returns

Tensor
    An tensor of the specified shape filled with random values.
