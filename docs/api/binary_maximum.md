# maximum

## Signature

```python
nabla.maximum(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Computes the element-wise maximum of two arrays.

This function compares two arrays element-wise and returns a new array
containing the larger of the two elements at each position. It supports
broadcasting.

Parameters
----------
x : Array | float | int
The first input array or scalar.
y : Array | float | int
The second input array or scalar. Must be broadcastable to the same
shape as `x`.

Returns
-------
Array
An array containing the element-wise maximum of `x` and `y`.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([1, 5, 2])
>>> y = nb.array([2, 3, 6])
>>> nb.maximum(x, y)
Array([2, 5, 6], dtype=int32)

