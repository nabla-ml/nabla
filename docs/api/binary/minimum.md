# minimum

## Signature

```python
nabla.minimum(x: 'Array | float | int', y: 'Array | float | int') -> 'Array'
```

## Description

Computes the element-wise minimum of two arrays.

This function compares two arrays element-wise and returns a new array
containing the smaller of the two elements at each position. It supports
broadcasting.

## Parameters

- **`x`** (`Array | float | int`): The first input array or scalar.

- **`y`** (`Array | float | int`): The second input array or scalar. Must be broadcastable to the same shape as `x`.

## Returns

- `Array`: An array containing the element-wise minimum of `x` and `y`.

## Examples

```python
import nabla as nb
x = nb.array([1, 5, 2])
y = nb.array([2, 3, 6])
nb.minimum(x, y)
Array([1, 3, 2], dtype=int32)
```
