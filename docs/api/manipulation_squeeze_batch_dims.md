# squeeze_batch_dims

## Signature

```python
nabla.squeeze_batch_dims(arg: nabla.core.array.Array, axes: list[int] | None = None) -> nabla.core.array.Array
```

## Description

Squeeze array by removing batch dimensions of size 1.


## Parameters

arg: Input array
axes: List of batch dimension axes to squeeze. If None, returns array unchanged.


## Returns

Array with specified batch dimensions of size 1 removed

