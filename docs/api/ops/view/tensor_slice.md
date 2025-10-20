# tensor_slice

## Signature

```python
nabla.tensor_slice(arg: nabla.core.tensor.Tensor, slices: list[slice], squeeze_axes: list[int] | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.view`

## Description

Slice an tensor along specified dimensions.

## Parameters

- **`arg`** (`Input tensor to slice`): slices: List of slice objects defining the slicing for each dimension squeeze_axes: List of axes that should be squeezed (for JAX compatibility)

## Returns

- `Sliced tensor`: 
