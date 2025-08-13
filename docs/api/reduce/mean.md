# mean

## Signature

```python
nabla.mean(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None', keep_dims: 'bool') -> 'Array'
```

## Description

Computes the arithmetic mean of array elements over given axes.

This function calculates the average of an array's elements along the
specified axes. If no axes are provided, the mean of all elements in the
array is calculated.

## Parameters

- **`arg`** (`Array`): The input array for which to compute the mean.

- **`axes`** (`int | list[int] | tuple[int, ...] | None, optional`): The axis or axes along which to compute the mean. If None (the default), the mean is computed over all axes, resulting in a scalar array.

- **`keep_dims`** (`bool, optional`): If True, the axes which are reduced are left in the result as dimensions with size one. This allows the result to broadcast correctly against the original array. Defaults to False.

## Returns

- `Array`: An array containing the mean values, typically of a floating-point dtype.

## Examples

```python
>>> import nabla as nb
>>> x = nb.array([[1, 2, 3], [4, 5, 6]])

Compute the mean of all elements:
>>> nb.mean(x)
Array([3.5], dtype=float32)

Compute the mean along an axis:
>>> nb.mean(x, axes=0)
Array([2.5, 3.5, 4.5], dtype=float32)

Compute the mean along an axis and keep dimensions:
>>> nb.mean(x, axes=1, keep_dims=True)
Array([[2.],
       [5.]], dtype=float32)
```
