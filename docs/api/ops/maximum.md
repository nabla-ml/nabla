# maximum

## Signature

```python
nabla.maximum(x: 'Tensor | float | int', y: 'Tensor | float | int') -> 'Tensor'
```

**Source**: `nabla.ops.binary`

## Description

Computes the element-wise maximum of two tensors.

This function compares two tensors element-wise and returns a new tensor
containing the larger of the two elements at each position. It supports
broadcasting.

## Parameters

- **`x`** (`Tensor | float | int`): The first input tensor or scalar.

- **`y`** (`Tensor | float | int`): The second input tensor or scalar. Must be broadcastable to the same shape as `x`.

## Returns

Tensor
    An tensor containing the element-wise maximum of `x` and `y`.

## Examples

```python
import nabla as nb
x = nb.tensor([1, 5, 2])
y = nb.tensor([2, 3, 6])
nb.maximum(x, y)
```
