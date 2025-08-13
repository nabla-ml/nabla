# sum

## Signature

```python
nabla.sum(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None', keep_dims: 'bool') -> 'Array'
```

## Description

Calculates the sum of array elements over given axes.

This function reduces an array by summing its elements along the
specified axes. If no axes are provided, the sum of all elements in the
array is calculated.

## Parameters

- **`arg`** (`Array`): The input array to be summed.

- **`axes`** (`int | list[int] | tuple[int, ...] | None, optional`): The axis or axes along which to perform the sum. If None (the default), the sum is performed over all axes, resulting in a scalar array.

- **`keep_dims`** (`bool, optional`): If True, the axes which are reduced are left in the result as dimensions with size one. This allows the result to broadcast correctly against the original array. Defaults to False.

## Returns

- `Array`: An array containing the summed values.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([[1, 2, 3], [4, 5, 6]])

Sum all elements:
>>> nb.sum(x)
Array([21], dtype=int32)

Sum along an axis:
>>> nb.sum(x, axes=0)
Array([5, 7, 9], dtype=int32)

Sum along an axis and keep dimensions:
>>> nb.sum(x, axes=1, keep_dims=True)
Array([[ 6],
       [15]], dtype=int32)
```
