# pad

## Signature

```python
nabla.pad(arg: nabla.core.array.Array, slices: list[slice], target_shape: tuple[int, ...]) -> nabla.core.array.Array
```

## Description

Place a smaller array into a larger zero-filled array at the location specified by slices.

This is the inverse operation of array slicing - given slices, a small array, and target shape,
it creates a larger array where the small array is placed at the sliced location
and everything else is zero.


## Parameters

arg: Input array (the smaller array to be placed)
slices: List of slice objects defining where to place the array
target_shape: The shape of the output array


## Returns

Larger array with input placed at sliced location, zeros elsewhere

