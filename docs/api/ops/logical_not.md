# logical_not

## Signature

```python
nabla.logical_not(arg: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.unary`

## Description

Computes the element-wise logical NOT of a boolean tensor.

This function inverts the boolean value of each element in the input tensor.
Input tensors of non-boolean types will be cast to boolean first.

## Parameters

- **`arg`** (`Tensor`): The input boolean tensor.

## Returns

- `Tensor`: A boolean tensor containing the inverted values.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([True, False, True])
>>> nb.logical_not(x)
Tensor([False,  True, False], dtype=bool)
```
