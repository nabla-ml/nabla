# incr_batch_dim_ctr

## Signature

```python
nabla.incr_batch_dim_ctr(arg: 'Tensor') -> 'Tensor'
```

## Description

Moves the leading axis from `shape` to `batch_dims`. (Internal use)

This is an internal-use function primarily for developing function
transformations like `vmap`. It re-interprets the first dimension of the
tensor's logical shape as a new batch dimension.

## Parameters

- **`arg`** (`Tensor`): The input tensor.

## Returns

- `Tensor`: A new tensor with an additional batch dimension.
