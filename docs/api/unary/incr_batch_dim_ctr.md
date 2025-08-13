# incr_batch_dim_ctr

## Signature

```python
nabla.incr_batch_dim_ctr(arg: 'Array') -> 'Array'
```

## Description

Moves the leading axis from `shape` to `batch_dims`. (Internal use)

This is an internal-use function primarily for developing function
transformations like `vmap`. It re-interprets the first dimension of the
array's logical shape as a new batch dimension.

## Parameters

- **`arg`** (`Array`): The input array.

## Returns

- `Array`: A new array with an additional batch dimension.
