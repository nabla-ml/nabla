# unsqueeze_batch_dims

## Signature

```python
nabla.unsqueeze_batch_dims(arg: nabla.core.tensor.Tensor, axes: list[int] | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.view`

## Description

Unsqueeze tensor by adding batch dimensions of size 1.

## Parameters

- **`arg`** (`Input tensor`): axes: List of positions where to insert batch dimensions of size 1. If None, returns tensor unchanged.

## Returns

- `Tensor with batch dimensions of size 1 added at specified positions`: 
