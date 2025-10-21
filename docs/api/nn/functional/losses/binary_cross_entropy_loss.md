# binary_cross_entropy_loss

## Signature

```python
nabla.nn.binary_cross_entropy_loss(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor, eps: float = 1e-07) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.losses.classification`

Compute binary cross-entropy loss.

Args:
    predictions: Model predictions (after sigmoid) [batch_size]
    targets: Binary targets (0 or 1) [batch_size]
    eps: Small constant for numerical stability

Returns:
    Scalar loss value

