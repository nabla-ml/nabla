# sqrt

## Signature

```python
nabla.sqrt(arg: 'Array') -> 'Array'
```

## Description

Computes the element-wise non-negative square root of an array.

This function is implemented as `nabla.pow(arg, 0.5)` to ensure it is
compatible with the automatic differentiation system.

## Parameters

- **`arg`** (`Array`): The input array. All elements must be non-negative.

## Returns

- `Array`: An array containing the square root of each element.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([0.0, 4.0, 9.0])
>>> nb.sqrt(x)
Array([0., 2., 3.], dtype=float32)
```
