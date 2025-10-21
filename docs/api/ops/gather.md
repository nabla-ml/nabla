# gather

## Signature

```python
nabla.gather(input_tensor: nabla.core.tensor.Tensor, indices: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.indexing`

## Description

Selects elements from an input tensor using indices along a specified axis.

This function is analogous to `numpy.take_along_axis`. It selects elements
from `input_tensor` at the positions specified by `indices`.

## Parameters

- **`input_tensor`** (`Tensor`): The source tensor from which to gather values.

- **`indices`** (`Tensor`): The tensor of indices to gather. Must be an integer-typed tensor.

- **`axis`** (`int, optional`): The axis along which to gather. A negative value counts from the last dimension. Defaults to -1.

## Returns

Tensor
    A new tensor containing the elements of `input_tensor` at the given
    `indices`.

## Examples

```python
import nabla as nb
x = nb.tensor([[10, 20, 30], [40, 50, 60]])
indices = nb.tensor([[0, 2], [1, 0]])
# Gather along axis 1
nb.gather(x, indices, axis=1)
```

```python
# Gather along axis 0
indices = nb.tensor([[0, 1, 0]])
nb.gather(x, indices, axis=0)
```
