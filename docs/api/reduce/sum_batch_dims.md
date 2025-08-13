# sum_batch_dims

## Signature

```python
nabla.sum_batch_dims(arg: 'Array', axes: 'int | list[int] | tuple[int, ...] | None', keep_dims: 'bool') -> 'Array'
```

## Description

Calculates the sum of array elements over given batch dimension axes.

This function is specialized for reducing batch dimensions, which are
used in function transformations like `vmap`. It operates on the
`batch_dims` of an array, leaving the standard `shape` unaffected.

## Parameters

- **`arg`** (`Array`): The input array with batch dimensions.

- **`axes`** (`int | list[int] | tuple[int, ...] | None, optional`): The batch dimension axis or axes to sum over. If None, sums over all batch dimensions.

- **`keep_dims`** (`bool, optional`): If True, the reduced batch axes are kept with size one. Defaults to False.

## Returns

- `Array`: An array with specified batch dimensions reduced by the sum operation.
