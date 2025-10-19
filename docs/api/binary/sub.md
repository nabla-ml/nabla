# sub

## Signature

```python
nabla.sub(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

## Description

Subtracts two tensors element-wise.

This function performs element-wise subtraction on two tensors. It supports
broadcasting, allowing tensors of different shapes to be combined as long
as their shapes are compatible. This function also provides the
implementation of the `-` operator for Nabla tensors.

## Parameters

- **`x`** (`Tensor | float | int`): The first input tensor or scalar (the minuend).

- **`y`** (`Tensor | float | int`): The second input tensor or scalar (the subtrahend). Must be broadcastable to the same shape as `x`.

## Returns

- `Tensor`: An tensor containing the result of the element-wise subtraction.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([10, 20, 30])
>>> y = nb.tensor([1, 2, 3])
>>> nb.sub(x, y)
Tensor([ 9, 18, 27], dtype=int32)

>>> x - y
Tensor([ 9, 18, 27], dtype=int32)
```
