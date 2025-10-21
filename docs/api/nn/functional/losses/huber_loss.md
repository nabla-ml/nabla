# huber_loss

## Signature

```python
nabla.nn.huber_loss(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor, delta: float = 1.0) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.losses.regression`

Compute Huber loss (smooth L1 loss).

Args:
    predictions: Predicted values of shape (batch_size, ...)
    targets: Target values of shape (batch_size, ...)
    delta: Threshold for switching between L1 and L2 loss

Returns:
    Scalar loss value

