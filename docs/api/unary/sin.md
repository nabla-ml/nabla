# sin

## Signature

```python
nabla.sin(arg: 'Array', dtype: 'DType | None') -> 'Array'
```

## Description

Computes the element-wise sine of an array.

## Parameters

- **`arg`** (`Array`): The input array. Input is expected to be in radians.

- **`dtype`** (`DType | None, optional`): If provided, the output array will be cast to this data type.

## Returns

- `Array`: An array containing the sine of each element in the input.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([0, 1.5707963, 3.1415926])
>>> nb.sin(x)
Array([0.0000000e+00, 1.0000000e+00, -8.7422780e-08], dtype=float32)
```
