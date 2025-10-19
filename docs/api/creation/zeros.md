# zeros

## Signature

```python
nabla.zeros(shape: 'Shape', dtype: 'DType', device: 'Device', batch_dims: 'Shape', traced: 'bool') -> 'Tensor'
```

## Description

Creates an tensor of a given shape filled with zeros.

## Parameters

- **`shape`** (`Shape`): The shape of the new tensor, e.g., `(2, 3)` or `(5,)`.

- **`dtype`** (`DType, optional`): The desired data type for the tensor. Defaults to DType.float32.

- **`device`** (`Device, optional`): The device to place the tensor on. Defaults to the CPU.

- **`batch_dims`** (`Shape, optional`): Specifies leading dimensions to be treated as batch dimensions. Defaults to an empty tuple.

- **`traced`** (`bool, optional`): Whether the operation should be traced in the graph. Defaults to False.

## Returns

- `Tensor`: An tensor of the specified shape and dtype, filled with zeros.

## Examples

```pycon
>>> import nabla as nb
>>> # Create a 2x3 matrix of zeros
>>> nb.zeros((2, 3), dtype=nb.DType.int32)
Tensor([[0, 0, 0],
       [0, 0, 0]], dtype=int32)
```
