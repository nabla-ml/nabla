# where

## Signature

```python
nabla.where(condition: 'Array', x: 'Array', y: 'Array') -> 'Array'
```

## Description

Selects elements from two arrays based on a condition.

This function returns an array with elements chosen from `x` where the
corresponding element in `condition` is True, and from `y` otherwise.
The function supports broadcasting among the three input arrays.

## Parameters

- **`condition`** (`Array`): A boolean array. Where True, yield `x`, otherwise yield `y`.

- **`x`** (`Array`): The array from which to take values when `condition` is True.

- **`y`** (`Array`): The array from which to take values when `condition` is False.

## Returns

- `Array`: An array with elements from `x` and `y`, depending on `condition`.

## Examples

```pycon
>>> import nabla as nb
>>> condition = nb.array([True, False, True])
>>> x = nb.array([1, 2, 3])
>>> y = nb.array([10, 20, 30])
>>> nb.where(condition, x, y)
Array([1, 20, 3], dtype=int32)

Broadcasting example:
>>> nb.where(nb.array([True, False]), nb.array(5), nb.array([10, 20]))
Array([5, 20], dtype=int32)
```
