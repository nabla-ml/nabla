# abs

## Signature

```python
nabla.abs(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

## Description

Computes the element-wise absolute value of an tensor.

## Parameters

- **`arg`** (`Tensor`): The input tensor.

## Returns

Tensor
    An tensor containing the absolute value of each element.

## Examples

```python
import nabla as nb
x = nb.tensor([-1.5, 0.0, 2.5])
nb.abs(x)
```
