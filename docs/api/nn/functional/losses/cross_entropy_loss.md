# cross_entropy_loss

## Signature

```python
nabla.nn.cross_entropy_loss(logits: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.losses.classification`

Compute cross-entropy loss between logits and targets.

Args:
    logits: Raw model outputs (before softmax) [batch_size, num_classes]
    targets: One-hot encoded targets [batch_size, num_classes]
    axis: Axis along which to compute softmax

Returns:
    Scalar loss value

