# permute_batch_dims

## Signature

```python
nabla.permute_batch_dims(input_tensor: nabla.core.tensor.Tensor, axes: tuple[int, ...]) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.view`

Permute (reorder) the batch dimensions of an tensor.

This operation reorders the batch_dims of an Tensor according to the given axes,
similar to how regular permute works on shape dimensions. The shape dimensions
remain unchanged.

Parameters
----------
    input_tensor: Input tensor with batch dimensions to permute
    axes: Tuple specifying the new order of batch dimensions.
          All indices should be negative and form a permutation.

Returns
-------
    Tensor with batch dimensions reordered according to axes

Examples
--------
    >>> import nabla as nb
    >>> # Tensor with batch_dims=(2, 3, 4) and shape=(5, 6)
    >>> x = nb.ones((5, 6))
    >>> x.batch_dims = (2, 3, 4)  # Simulated for example
    >>> y = permute_batch_dims(x, (-1, -3, -2))  # Reorder as (4, 2, 3)
    >>> # Result has batch_dims=(4, 2, 3) and shape=(5, 6)

