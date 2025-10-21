# precision

## Signature

```python
nabla.nn.precision(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor, num_classes: int, class_idx: int = 0) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.metrics`

Compute precision for a specific class.

Precision = TP / (TP + FP)

Args:
    predictions: Model predictions (logits) [batch_size, num_classes]
    targets: True labels (sparse) [batch_size]
    num_classes: Total number of classes
    class_idx: Class index to compute precision for

Returns:
    Scalar precision value for the specified class

