# floordiv

## Signature

```python
nabla.floordiv(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Performs element-wise floor division on two arrays.

Floor division is equivalent to `floor(x / y)`, rounding the result
towards negative infinity. This matches the behavior of Python's `//`
operator, which this function implements for Nabla arrays.

## Parameters

- **`x`** (`Array | float | int`): The first input array or scalar (the dividend).

- **`y`** (`Array | float | int`): The second input array or scalar (the divisor). Must be broadcastable to the same shape as `x`.

## Returns

- `Array`: An array containing the result of the element-wise floor division.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([10, -10, 9])
>>> y = nb.array([3, 3, 3])
>>> nb.floordiv(x, y)
Array([ 3, -4,  3], dtype=int32)

>>> x // y
Array([ 3, -4,  3], dtype=int32)
```
