# sum

## Signature

```python
nabla.sum(arg: 'Tensor', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.reduce`

## Description

Calculates the sum of tensor elements over given axes.

This function reduces an tensor by summing its elements along the
specified axes. If no axes are provided, the sum of all elements in the
tensor is calculated.

## Parameters

- **`arg`** (`Tensor`): The input tensor to be summed.

- **`axes`** (`int | list[int] | tuple[int, ...] | None, optional`): The axis or axes along which to perform the sum. If None (the default), the sum is performed over all axes, resulting in a scalar tensor.

- **`keep_dims`** (`bool, optional`): If True, the axes which are reduced are left in the result as dimensions with size one. This allows the result to broadcast correctly against the original tensor. Defaults to False.

## Returns

- `Tensor`: An tensor containing the summed values.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([[1, 2, 3], [4, 5, 6]])

Sum all elements:
>>> nb.sum(x)
Tensor([21], dtype=int32)

Sum along an axis:
>>> nb.sum(x, axes=0)
Tensor([5, 7, 9], dtype=int32)

Sum along an axis and keep dimensions:
>>> nb.sum(x, axes=1, keep_dims=True)
Tensor([[ 6],
       [15]], dtype=int32)
```
