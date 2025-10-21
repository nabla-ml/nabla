# mean_absolute_error_metric

## Signature

```python
nabla.nn.mean_absolute_error_metric(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.metrics`

Compute MAE metric for regression tasks.

Args:
    predictions: Model predictions [batch_size, ...]
    targets: True targets [batch_size, ...]

Returns:
    Scalar MAE value

