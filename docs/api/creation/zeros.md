# zeros

## Signature

```python
nabla.zeros(shape: 'Shape', dtype: 'DType', device: 'Device', batch_dims: 'Shape', traced: 'bool') -> 'Array'
```

## Description

Creates an array of a given shape filled with zeros.

## Parameters

- **`shape`** (`Shape`): The shape of the new array, e.g., `(2, 3)` or `(5,)`.

- **`dtype`** (`DType, optional`): The desired data type for the array. Defaults to DType.float32.

- **`device`** (`Device, optional`): The device to place the array on. Defaults to the CPU.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

## Returns

- `Array`: An array of the specified shape and dtype, filled with zeros.

## Examples

```python
>>> import nabla as nb
>>> # Create a 2x3 matrix of zeros
>>> nb.zeros((2, 3), dtype=nb.DType.int32)
Array([[0, 0, 0],
       [0, 0, 0]], dtype=int32)
```
