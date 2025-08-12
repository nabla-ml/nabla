# sub

## Signature

```python
nabla.sub(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Subtracts two arrays element-wise.

This function performs element-wise subtraction on two arrays. It supports
broadcasting, allowing arrays of different shapes to be combined as long
as their shapes are compatible. This function also provides the
implementation of the `-` operator for Nabla arrays.

Parameters
----------
x : Array | float | int
The first input array or scalar (the minuend).
y : Array | float | int
The second input array or scalar (the subtrahend). Must be
broadcastable to the same shape as `x`.

Returns
-------
Array
An array containing the result of the element-wise subtraction.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([10, 20, 30])
>>> y = nb.array([1, 2, 3])
>>> nb.sub(x, y)
Array([ 9, 18, 27], dtype=int32)

>>> x - y
Array([ 9, 18, 27], dtype=int32)

## See Also

- {doc}`add <binary_add>` - Addition
- {doc}`mul <binary_mul>` - Multiplication
- {doc}`div <binary_div>` - Division

