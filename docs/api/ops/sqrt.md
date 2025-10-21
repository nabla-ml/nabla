# sqrt

## Signature

```python
nabla.sqrt(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

## Description

Computes the element-wise non-negative square root of an tensor.

This function is implemented as `nabla.pow(arg, 0.5)` to ensure it is
compatible with the automatic differentiation system.

## Parameters

- **`arg`** (`Tensor`): The input tensor. All elements must be non-negative.

## Returns

Tensor
    An tensor containing the square root of each element.

## Examples

```python
import nabla as nb
x = nb.tensor([0.0, 4.0, 9.0])
nb.sqrt(x)
```
