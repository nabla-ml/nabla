# r_squared

## Signature

```python
nabla.nn.r_squared(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.metrics`

Compute R-squared (coefficient of determination) for regression tasks.

R² = 1 - (SS_res / SS_tot)
where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - y_mean)²

Args:
    predictions: Model predictions [batch_size, ...]
    targets: True targets [batch_size, ...]

Returns:
    Scalar R² value

