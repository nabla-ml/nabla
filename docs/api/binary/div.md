# div

## Signature

```python
nabla.div(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Divides two arrays element-wise.

This function performs element-wise (true) division on two arrays. It
supports broadcasting, allowing arrays of different shapes to be combined
as long as their shapes are compatible. This function also provides the
implementation of the `/` operator for Nabla arrays.

## Parameters

- **`x`** (`Array | float | int`): The first input array or scalar (the dividend).

- **`y`** (`Array | float | int`): The second input array or scalar (the divisor). Must be broadcastable to the same shape as `x`.

## Returns

- `Array`: An array containing the result of the element-wise division. The result is typically a floating-point array.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([10, 20, 30])
>>> y = nb.array([2, 5, 10])
>>> nb.div(x, y)
Array([5., 4., 3.], dtype=float32)

>>> x / y
Array([5., 4., 3.], dtype=float32)
```
