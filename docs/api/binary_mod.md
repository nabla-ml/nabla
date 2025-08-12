# mod

## Signature

```python
nabla.mod(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Computes the element-wise remainder of division.

This function calculates the remainder of `x / y` element-wise. The
sign of the result follows the sign of the divisor `y`. It provides the
implementation of the `%` operator for Nabla arrays.

Parameters
----------
x : Array | float | int
The dividend array or scalar.
y : Array | float | int
The divisor array or scalar. Must be broadcastable to the same shape
as `x`.

Returns
-------
Array
An array containing the element-wise remainder.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([10, -10, 9])
>>> y = nb.array([3, 3, -3])
>>> nb.mod(x, y)
Array([ 1,  2, -0], dtype=int32)

>>> x % y
Array([ 1,  2, -0], dtype=int32)

