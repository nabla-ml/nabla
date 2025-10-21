# cos

## Signature

```python
nabla.cos(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

## Description

Computes the element-wise cosine of an tensor.

## Parameters

- **`arg`** (`Tensor`): The input tensor. Input is expected to be in radians.

## Returns

- `Tensor`: An tensor containing the cosine of each element in the input.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([0, 1.5707963, 3.1415926])
>>> nb.cos(x)
Tensor([ 1.000000e+00, -4.371139e-08, -1.000000e+00], dtype=float32)
```
