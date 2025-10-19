# unsqueeze_batch_dims

## Signature

```python
nabla.unsqueeze_batch_dims(arg: 'Tensor', axes: 'list[int] | None') -> 'Tensor'
```

## Description

Unsqueeze tensor by adding batch dimensions of size 1.

## Parameters

- **`arg`** (`Input tensor`): axes: List of positions where to insert batch dimensions of size 1. If None, returns tensor unchanged.

## Returns

- `Tensor with batch dimensions of size 1 added at specified positions`: 
