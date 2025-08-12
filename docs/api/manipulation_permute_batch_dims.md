# permute_batch_dims

## Signature

```python
nabla.permute_batch_dims(input_array: nabla.core.array.Array, axes: tuple[int, ...]) -> nabla.core.array.Array
```

## Description

Permute (reorder) the batch dimensions of an array.

This operation reorders the batch_dims of an Array according to the given axes,
similar to how regular permute works on shape dimensions. The shape dimensions
remain unchanged.

Parameters
----------
input_array: Input array with batch dimensions to permute
axes: Tuple specifying the new order of batch dimensions.
All indices should be negative and form a permutation.

Returns
-------
Array with batch dimensions reordered according to axes

Examples
--------
>>> import nabla as nb
>>> # Array with batch_dims=(2, 3, 4) and shape=(5, 6)
>>> x = nb.ones((5, 6))
>>> x.batch_dims = (2, 3, 4)  # Simulated for example
>>> y = permute_batch_dims(x, (-1, -3, -2))  # Reorder as (4, 2, 3)
>>> # Result has batch_dims=(4, 2, 3) and shape=(5, 6)

