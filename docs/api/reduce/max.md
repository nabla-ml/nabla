# max

## Signature

```python
nabla.max(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None', keep_dims: 'bool') -> 'Array'
```

## Description

Finds the maximum value of array elements over given axes.

This function reduces an array by finding the maximum element along the
specified axes. If no axes are provided, the maximum of all elements in the
array is returned.

## Parameters

- **`arg`** (`Array`): The input array.

- **`axes`** (`int | list[int] | tuple[int, ...] | None, optional`): The axis or axes along which to find the maximum. If None (the default), the maximum is found over all axes, resulting in a scalar array.

- **`keep_dims`** (`bool, optional`): If True, the axes which are reduced are left in the result as dimensions with size one. This allows the result to broadcast correctly against the original array. Defaults to False.

## Returns

- `Array`: An array containing the maximum values.

## Examples

```python
import nabla as nb
x = nb.array([[1, 5, 2], [4, 3, 6]])

Find the maximum of all elements:
nb.max(x)
Array([6], dtype=int32)

Find the maximum along an axis:
nb.max(x, axes=1)
Array([5, 6], dtype=int32)

Find the maximum along an axis and keep dimensions:
nb.max(x, axes=0, keep_dims=True)
Array([[4, 5, 6]], dtype=int32)
```
