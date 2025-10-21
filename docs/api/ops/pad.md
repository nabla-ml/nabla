# pad

## Signature

```python
nabla.pad(arg: nabla.core.tensor.Tensor, slices: list[slice], target_shape: tuple[int, ...]) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.view`

## Description

Place a smaller tensor into a larger zero-filled tensor at the location specified by slices.

This is the inverse operation of tensor slicing - given slices, a small tensor, and target shape,
it creates a larger tensor where the small tensor is placed at the sliced location
and everything else is zero.

## Parameters

- **`arg`** (`Input tensor (the smaller tensor to be placed)`): 

- **`slices`** (`List of slice objects defining where to place the tensor`): 

- **`target_shape`** (`The shape of the output tensor`): 

## Returns

Larger tensor with input placed at sliced location, zeros elsewhere
