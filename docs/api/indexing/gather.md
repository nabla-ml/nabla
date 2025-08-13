# gather

## Signature

```python
nabla.gather(input_array: 'Array', indices: 'Array', axis: 'int') -> 'Array'
```

## Description

Selects elements from an input array using indices along a specified axis.

This function is analogous to `numpy.take_along_axis`. It selects elements
from `input_array` at the positions specified by `indices`.

## Parameters

- **`input_array`** (`Array`): The source array from which to gather values.

- **`indices`** (`Array`): The array of indices to gather. Must be an integer-typed array.

- **`axis`** (`int, optional`): The axis along which to gather. A negative value counts from the last dimension. Defaults to -1.

## Returns

- `Array`: A new array containing the elements of `input_array` at the given `indices`.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([[10, 20, 30], [40, 50, 60]])
>>> indices = nb.array([[0, 2], [1, 0]])
>>> # Gather along axis 1
>>> nb.gather(x, indices, axis=1)
Array([[10, 30],
       [50, 40]], dtype=int32)

>>> # Gather along axis 0
>>> indices = nb.array([[0, 1, 0]])
>>> nb.gather(x, indices, axis=0)
Array([[10, 50, 30]], dtype=int32)
```
