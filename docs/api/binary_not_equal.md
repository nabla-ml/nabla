# not_equal

## Signature

```python
nabla.not_equal(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Performs element-wise comparison `x != y`.

This function compares two arrays element-wise, returning a boolean array
indicating where elements of `x` are not equal to elements of `y`. It
supports broadcasting and provides the implementation of the `!=` operator
for Nabla arrays.

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
A boolean array containing the result of the element-wise inequality
comparison.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([1, 2, 3])
>>> y = nb.array([1, 5, 3])
>>> nb.not_equal(x, y)
Array([False,  True, False], dtype=bool)

>>> x != y
Array([False,  True, False], dtype=bool)

