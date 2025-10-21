# softmax_cross_entropy_loss

## Signature

```python
nabla.nn.softmax_cross_entropy_loss(logits: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.losses.classification`

Compute softmax cross-entropy loss (numerically stable).

This is equivalent to cross_entropy_loss but more numerically stable
by combining softmax and cross-entropy computations.

Args:
    logits: Raw model outputs [batch_size, num_classes]
    targets: One-hot encoded targets [batch_size, num_classes]
    axis: Axis along which to compute softmax

Returns:
    Scalar loss value

