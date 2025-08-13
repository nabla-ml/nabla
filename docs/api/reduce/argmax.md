# argmax

## Signature

```python
nabla.argmax(arg: 'Array', axes: 'int | None', keep_dims: 'bool') -> 'Array'
```

## Description

Finds the indices of maximum array elements over a given axis.

This function returns the indices of the maximum values along an axis. If
multiple occurrences of the maximum value exist, the index of the first
occurrence is returned.

## Parameters

- **`arg`** (`Array`): The input array.

- **`axes`** (`int | None, optional`): The axis along which to find the indices of the maximum values. If None (the default), the array is flattened before finding the index of the overall maximum value.

- **`keep_dims`** (`bool, optional`): If True, the axis which is reduced is left in the result as a dimension with size one. This is not supported when `axes` is None. Defaults to False.

## Returns

- `Array`: An array of `int64` integers containing the indices of the maximum elements.

## Examples

```python
import nabla as nb
x = nb.array([1, 5, 2, 5])
nb.argmax(x)
Array(1, dtype=int64)

y = nb.array([[1, 5, 2], [4, 3, 6]])
nb.argmax(y, axes=1)
Array([1, 2], dtype=int64)

nb.argmax(y, axes=0, keep_dims=True)
Array([[1, 0, 1]], dtype=int64)
```
