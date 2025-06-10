# array_slice

## Signature

```python
nabla.array_slice(arg: nabla.core.array.Array, slices: list[slice], squeeze_axes: list[int] = None) -> nabla.core.array.Array
```

## Description

Slice an array along specified dimensions.


## Parameters

arg: Input array to slice
slices: List of slice objects defining the slicing for each dimension
squeeze_axes: List of axes that should be squeezed (for JAX compatibility)


## Returns

Sliced array

