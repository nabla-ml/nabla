# cast

## Signature

```python
nabla.cast(arg: 'Array', dtype: 'DType') -> 'Array'
```

## Description

Casts an array to a specified data type.

This function creates a new array with the same shape as the input but
with the specified data type (`dtype`).

## Parameters

- **`arg`** (`Array`): The input array to be cast.

- **`dtype`** (`DType`): The target Nabla data type (e.g., `nb.float32`, `nb.int32`).

## Returns

- `Array`: A new array with the elements cast to the specified `dtype`.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.array([1, 2, 3])
>>> x.dtype
int32
>>> y = nb.cast(x, nb.float32)
>>> y
Array([1., 2., 3.], dtype=float32)
```
