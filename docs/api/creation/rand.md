# rand

## Signature

```python
nabla.rand(shape: 'Shape', dtype: 'DType', lower: 'float', upper: 'float', device: 'Device', seed: 'int', batch_dims: 'Shape', traced: 'bool') -> 'Array'
```

## Description

Creates an array with uniformly distributed random values.

The values are drawn from a continuous uniform distribution over the
interval `[lower, upper)`.

## Parameters

- **`shape`** (`Shape`): The shape of the output array.

- **`dtype`** (`DType, optional`): The desired data type for the array. Defaults to DType.float32.

- **`lower`** (`float, optional`): The lower boundary of the output interval. Defaults to 0.0.

- **`upper`** (`float, optional`): The upper boundary of the output interval. Defaults to 1.0.

- **`device`** (`Device, optional`): The device to place the array on. Defaults to the CPU.

- **`seed`** (`int, optional`): The seed for the random number generator. Defaults to 0.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

## Returns

- `Array`: An array of the specified shape filled with random values.
