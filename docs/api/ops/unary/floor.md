# floor

## Signature

```python
nabla.floor(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

## Description

Computes the element-wise floor of an tensor.

The floor of a scalar `x` is the largest integer `i` such that `i <= x`.
This function is not differentiable and its gradient is zero everywhere.

## Parameters

- **`arg`** (`Tensor`): The input tensor.

## Returns

- `Tensor`: An tensor containing the floor of each element.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([-1.7, -0.2, 0.2, 1.7])
>>> nb.floor(x)
Tensor([-2., -1.,  0.,  1.], dtype=float32)
```
