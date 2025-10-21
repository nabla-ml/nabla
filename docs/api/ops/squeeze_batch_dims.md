# squeeze_batch_dims

## Signature

```python
nabla.squeeze_batch_dims(arg: nabla.core.tensor.Tensor, axes: list[int] | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.view`

## Description

Squeeze tensor by removing batch dimensions of size 1.

## Parameters

- **`arg`** (`Input tensor`): 

- **`axes`** (`List of batch dimension axes to squeeze. If None, returns tensor unchanged.`): 

## Returns

Tensor with specified batch dimensions of size 1 removed
