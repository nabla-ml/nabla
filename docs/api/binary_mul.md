# mul

## Signature

```python
nabla.mul(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Multiplies two arrays element-wise.

This function performs element-wise multiplication on two arrays. It
supports broadcasting, allowing arrays of different shapes to be combined
as long as their shapes are compatible. This function also provides the
implementation of the `*` operator for Nabla arrays.

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
An array containing the result of the element-wise multiplication.

Examples
--------
>>> import nabla as nb
>>> x = nb.array([1, 2, 3])
>>> y = nb.array([4, 5, 6])
>>> nb.mul(x, y)
Array([4, 10, 18], dtype=int32)

>>> x * y
Array([4, 10, 18], dtype=int32)

## Examples

```python
import nabla as nb

# Element-wise multiplication
a = nb.array([1, 2, 3])
b = nb.array([4, 5, 6])
result = nb.mul(a, b)
print(result)  # [4, 10, 18]
```

## See Also

- {doc}`add <binary_add>` - Addition
- {doc}`div <binary_div>` - Division
- {doc}`pow <binary_pow>` - Exponentiation

