# tensor

## Signature

```python
nabla.tensor(data: 'list | np.ndarray | float | int', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), batch_dims: 'Shape' = (), traced: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

## Description

Creates an tensor from a Python list, NumPy tensor, or scalar.

This function is the primary way to create a Nabla tensor from existing
data. It converts the input data into a Nabla tensor on the specified
device and with the given data type.

## Parameters

- **`data`** (`list | np.ndarray | float | int`): The input data to convert to an tensor.

- **`dtype`** (`DType, optional`): The desired data type for the tensor. Defaults to DType.float32.

- **`device`** (`Device, optional`): The computational device where the tensor will be stored. Defaults to the CPU.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

## Returns

- `Tensor`: A new Nabla tensor containing the provided data.

## Examples

```pycon
>>> import nabla as nb
>>> import numpy as np
>>> # Create from a Python list
>>> nb.tensor([1, 2, 3])
Tensor([1, 2, 3], dtype=int32)

>>> # Create from a NumPy tensor
>>> np_arr = np.array([[4.0, 5.0], [6.0, 7.0]])
>>> nb.tensor(np_arr)
Tensor([[4., 5.],
       [6., 7.]], dtype=float32)

>>> # Create a scalar tensor
>>> nb.tensor(100, dtype=nb.DType.int64)
Tensor(100, dtype=int64)
```
