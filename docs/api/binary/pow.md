# pow

## Signature

```python
nabla.pow(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Computes `x` raised to the power of `y` element-wise.

This function calculates `x ** y` for each element in the input arrays.
It supports broadcasting and provides the implementation of the `**`
operator for Nabla arrays.

## Parameters

- **`x`** (`Array | float | int`): The base array or scalar.

- **`y`** (`Array | float | int`): The exponent array or scalar. Must be broadcastable to the same shape as `x`.

## Returns

- `Array`: An array containing the result of the element-wise power operation.

## Examples

```python
>>> import nabla as nb
>>> x = nb.array([1, 2, 3])
>>> y = nb.array([2, 3, 2])
>>> nb.pow(x, y)
Array([1, 8, 9], dtype=int32)

>>> x ** y
Array([1, 8, 9], dtype=int32)
```
