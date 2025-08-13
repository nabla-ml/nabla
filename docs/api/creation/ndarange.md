# ndarange

## Signature

```python
nabla.ndarange(shape: 'Shape', dtype: 'DType', device: 'Device', batch_dims: 'Shape', traced: 'bool') -> 'Array'
```

## Description

Creates an array of a given shape with sequential values.

The array is filled with values from 0 to N-1, where N is the total
number of elements (the product of the shape dimensions).

## Parameters

- **`shape`** (`Shape`): The shape of the output array.

- **`dtype`** (`DType, optional`): The desired data type for the array. Defaults to DType.float32.

- **`device`** (`Device, optional`): The device to place the array on. Defaults to the CPU.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

## Returns

- `Array`: An array of the specified shape containing values from 0 to N-1.

## Examples

```python
>>> import nabla as nb
>>> nb.ndarange((2, 3), dtype=nb.DType.int32)
Array([[0, 1, 2],
       [3, 4, 5]], dtype=int32)
```
