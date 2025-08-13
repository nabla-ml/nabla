# add

## Signature

```python
nabla.add(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Adds two arrays element-wise.

This function performs element-wise addition on two arrays. It supports
broadcasting, allowing arrays of different shapes to be combined as long
as their shapes are compatible. This function also provides the
implementation of the `+` operator for Nabla arrays.

## Parameters

- **`x`** (`Array | float | int`): The first input array or scalar.

- **`y`** (`Array | float | int`): The second input array or scalar. Must be broadcastable to the same shape as `x`.

## Returns

- `Array`: An array containing the result of the element-wise addition.

## Examples

```python
Calling `add` explicitly:

>>> import nabla as nb
>>> x = nb.array([1, 2, 3])
>>> y = nb.array([4, 5, 6])
>>> nb.add(x, y)
Array([5, 7, 9], dtype=int32)

Calling `add` via the `+` operator:

>>> x + y
Array([5, 7, 9], dtype=int32)

Broadcasting a scalar:

>>> x + 10
Array([11, 12, 13], dtype=int32)
```
