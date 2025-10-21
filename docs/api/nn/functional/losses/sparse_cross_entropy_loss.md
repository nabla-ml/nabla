# sparse_cross_entropy_loss

## Signature

```python
nabla.nn.sparse_cross_entropy_loss(logits: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.losses.classification`

Compute cross-entropy loss with integer targets.

Args:
    logits: Raw model outputs [batch_size, num_classes]
    targets: Integer class indices [batch_size]
    axis: Axis along which to compute softmax

Returns:
    Scalar loss value

