# array

## Signature

```python
nabla.array(data: 'list | np.ndarray | float | int', dtype: 'DType', device: 'Device', batch_dims: 'Shape', traced: 'bool') -> 'Array'
```

## Description

Creates an array from a Python list, NumPy array, or scalar.

This function is the primary way to create a Nabla array from existing
data. It converts the input data into a Nabla array on the specified
device and with the given data type.

## Parameters

- **`data`** (`list | np.ndarray | float | int`): The input data to convert to an array.

- **`dtype`** (`DType, optional`): The desired data type for the array. Defaults to DType.float32.

- **`device`** (`Device, optional`): The computational device where the array will be stored. Defaults to the CPU.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

## Returns

- `Array`: A new Nabla array containing the provided data.

## Examples

```python
>>> import nabla as nb
>>> import numpy as np
>>> # Create from a Python list
>>> nb.array([1, 2, 3])
Array([1, 2, 3], dtype=int32)
<BLANKLINE>
>>> # Create from a NumPy array
>>> np_arr = np.array([[4.0, 5.0], [6.0, 7.0]])
>>> nb.array(np_arr)
Array([[4., 5.],
       [6., 7.]], dtype=float32)
<BLANKLINE>
>>> # Create a scalar array
>>> nb.array(100, dtype=nb.DType.int64)
Array(100, dtype=int64)
```
