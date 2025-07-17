# permute

## Signature

```python
nabla.permute(input_array: nabla.core.array.Array, axes: tuple[int, ...]) -> nabla.core.array.Array
```

## Description

Permute (reorder) the dimensions of a tensor.


## Parameters

input_array: Input tensor
axes: Tuple specifying the new order of dimensions


## Returns

Tensor with reordered dimensions


## Examples

>>> x = nb.ones((2, 3, 4))  # shape (2, 3, 4)
>>> y = permute(x, (2, 0, 1))  # shape (4, 2, 3)
>>> # Dimension 2 -> position 0, dimension 0 -> position 1, dimension 1 -> position 2

