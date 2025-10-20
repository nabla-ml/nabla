# scatter

## Signature

```python
nabla.scatter(target_shape: tuple, indices: nabla.core.tensor.Tensor, values: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.indexing`

## Description

Updates an tensor of zeros with given values at specified indices.

This function creates an tensor of shape `target_shape` filled with zeros
and then places the `values` at the locations specified by `indices` along
the given `axis`. This operation is the inverse of `gather`.

## Parameters

- **`target_shape`** (`tuple`): The shape of the output tensor.

- **`indices`** (`Tensor`): An integer tensor specifying the indices to update.

- **`values`** (`Tensor`): The tensor of values to scatter into the new tensor.

- **`axis`** (`int, optional`): The axis along which to scatter. A negative value counts from the last dimension. Defaults to -1.

## Returns

- `Tensor`: A new tensor of shape `target_shape` with `values` scattered at the specified `indices`.

## Examples

```pycon
>>> import nabla as nb
>>> target_shape = (3, 4)
>>> indices = nb.tensor([0, 2, 1])
>>> values = nb.tensor([10, 20, 30])
>>> # Scatter values into a 1D target
>>> nb.scatter((4,), nb.tensor([0, 3, 1]), nb.tensor([1, 2, 3]))
Tensor([1, 3, 0, 2], dtype=int32)

>>> # Scatter rows into a 2D target along axis 0
>>> values_2d = nb.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
>>> nb.scatter(target_shape, indices, values_2d, axis=0)
Tensor([[1, 1, 1, 1],
       [3, 3, 3, 3],
       [2, 2, 2, 2]], dtype=int32)
```
