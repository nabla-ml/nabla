# logical_not

## Signature

```python
nabla.logical_not(arg: 'Array') -> 'Array'
```

## Description

Computes the element-wise logical NOT of a boolean array.

This function inverts the boolean value of each element in the input array.
Input arrays of non-boolean types will be cast to boolean first.

## Parameters

- **`arg`** (`Array`): The input boolean array.

## Returns

- `Array`: A boolean array containing the inverted values.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([True, False, True])
>>> nb.logical_not(x)
Array([False,  True, False], dtype=bool)
```
