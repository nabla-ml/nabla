# pearson_correlation

## Signature

```python
nabla.nn.pearson_correlation(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.metrics`

Compute Pearson correlation coefficient.

Args:
    predictions: Model predictions [batch_size, ...]
    targets: True targets [batch_size, ...]

Returns:
    Scalar correlation coefficient

