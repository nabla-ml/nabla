# squeeze_batch_dims

## Signature

```python
nabla.squeeze_batch_dims(arg: 'Tensor', axes: 'list[int] | None') -> 'Tensor'
```

## Description

Squeeze tensor by removing batch dimensions of size 1.

## Parameters

- **`arg`** (`Input tensor`): axes: List of batch dimension axes to squeeze. If None, returns tensor unchanged.

## Returns

- `Tensor with specified batch dimensions of size 1 removed`: 
