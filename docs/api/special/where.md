# where

## Signature

```python
nabla.where(condition: 'Tensor', x: 'Tensor', y: 'Tensor') -> 'Tensor'
```

## Description

Selects elements from two tensors based on a condition.

This function returns an tensor with elements chosen from `x` where the
corresponding element in `condition` is True, and from `y` otherwise.
The function supports broadcasting among the three input tensors.

## Parameters

- **`condition`** (`Tensor`): A boolean tensor. Where True, yield `x`, otherwise yield `y`.

- **`x`** (`Tensor`): The tensor from which to take values when `condition` is True.

- **`y`** (`Tensor`): The tensor from which to take values when `condition` is False.

## Returns

- `Tensor`: An tensor with elements from `x` and `y`, depending on `condition`.

## Examples

```pycon
>>> import nabla as nb
>>> condition = nb.tensor([True, False, True])
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([10, 20, 30])
>>> nb.where(condition, x, y)
Tensor([1, 20, 3], dtype=int32)

Broadcasting example:
>>> nb.where(nb.tensor([True, False]), nb.tensor(5), nb.tensor([10, 20]))
Tensor([5, 20], dtype=int32)
```
