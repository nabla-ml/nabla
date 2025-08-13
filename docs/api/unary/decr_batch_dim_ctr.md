# decr_batch_dim_ctr

## Signature

```python
nabla.decr_batch_dim_ctr(arg: 'Array') -> 'Array'
```

## Description

Moves the last `batch_dim` to be the leading axis of `shape`. (Internal use)

This is an internal-use function primarily for developing function
transformations like `vmap`. It re-interprets the last batch dimension
as the new first dimension of the array's logical shape.

## Parameters

- **`arg`** (`Array`): The input array.

## Returns

- `Array`: A new array with one fewer batch dimension.
