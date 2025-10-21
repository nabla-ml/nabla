# mean_absolute_error

## Signature

```python
nabla.nn.mean_absolute_error(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.losses.regression`

Compute mean absolute error loss.

Args:
    predictions: Predicted values of shape (batch_size, ...)
    targets: Target values of shape (batch_size, ...)

Returns:
    Scalar loss value

