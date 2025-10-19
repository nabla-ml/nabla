# cast

## Signature

```python
nabla.cast(arg: 'Tensor', dtype: 'DType') -> 'Tensor'
```

## Description

Casts an tensor to a specified data type.

This function creates a new tensor with the same shape as the input but
with the specified data type (`dtype`).

## Parameters

- **`arg`** (`Tensor`): The input tensor to be cast.

- **`dtype`** (`DType`): The target Nabla data type (e.g., `nb.float32`, `nb.int32`).

## Returns

- `Tensor`: A new tensor with the elements cast to the specified `dtype`.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> x.dtype
int32
>>> y = nb.cast(x, nb.float32)
>>> y
Tensor([1., 2., 3.], dtype=float32)
```
