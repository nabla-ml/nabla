# scatter

## Signature

```python
nabla.scatter(target_shape: 'tuple', indices: 'Array', values: 'Array', axis: 'int') -> 'Array'
```

## Description

Updates an array of zeros with given values at specified indices.

This function creates an array of shape `target_shape` filled with zeros
and then places the `values` at the locations specified by `indices` along
the given `axis`. This operation is the inverse of `gather`.

## Parameters

- **`target_shape`** (`tuple`): The shape of the output array.

- **`indices`** (`Array`): An integer array specifying the indices to update.

- **`values`** (`Array`): The array of values to scatter into the new array.

- **`axis`** (`int, optional`): The axis along which to scatter. A negative value counts from the last dimension. Defaults to -1.

## Returns

- `Array`: A new array of shape `target_shape` with `values` scattered at the specified `indices`.

## Examples

```python
import nabla as nb
target_shape = (3, 4)
indices = nb.array([0, 2, 1])
values = nb.array([10, 20, 30])
# Scatter values into a 1D target
nb.scatter((4,), nb.array([0, 3, 1]), nb.array([1, 2, 3]))
Array([1, 3, 0, 2], dtype=int32)

# Scatter rows into a 2D target along axis 0
values_2d = nb.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
nb.scatter(target_shape, indices, values_2d, axis=0)
Array([[1, 1, 1, 1],
       [3, 3, 3, 3],
       [2, 2, 2, 2]], dtype=int32)
```
