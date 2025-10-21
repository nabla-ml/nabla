# sum_batch_dims

## Signature

```python
nabla.sum_batch_dims(arg: 'Tensor', axes: 'int | list[int] | tuple[int, ...] | None' = None, keep_dims: 'bool' = False) -> 'Tensor'
```

**Source**: `nabla.ops.reduce`

## Description

Calculates the sum of tensor elements over given batch dimension axes.

This function is specialized for reducing batch dimensions, which are
used in function transformations like `vmap`. It operates on the
`batch_dims` of an tensor, leaving the standard `shape` unaffected.

## Parameters

- **`arg`** (`Tensor`): The input tensor with batch dimensions.

- **`axes`** (`int | list[int] | tuple[int, ...] | None, optional`): The batch dimension axis or axes to sum over. If None, sums over all batch dimensions.

- **`keep_dims`** (`bool, optional`): If True, the reduced batch axes are kept with size one. Defaults to False.

## Returns

- `Tensor`: An tensor with specified batch dimensions reduced by the sum operation.
