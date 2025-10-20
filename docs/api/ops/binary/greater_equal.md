# greater_equal

## Signature

```python
nabla.greater_equal(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

**Source**: `nabla.ops.binary`

## Description

Performs element-wise comparison `x >= y`.

This function compares two tensors element-wise, returning a boolean tensor
indicating where elements of `x` are greater than or equal to elements
of `y`. It supports broadcasting and provides the implementation of the
`>=` operator for Nabla tensors.

## Parameters

- **`x`** (`Tensor | float | int`): The first input tensor or scalar.

- **`y`** (`Tensor | float | int`): The second input tensor or scalar. Must be broadcastable to the same shape as `x`.

## Returns

- `Tensor`: A boolean tensor containing the result of the element-wise comparison.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([1, 5, 3])
>>> y = nb.tensor([2, 5, 1])
>>> nb.greater_equal(x, y)
Tensor([False,  True,  True], dtype=bool)

>>> x >= y
Tensor([False,  True,  True], dtype=bool)
```
