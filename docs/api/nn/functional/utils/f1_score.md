# f1_score

## Signature

```python
nabla.nn.f1_score(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor, num_classes: int, class_idx: int = 0) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.metrics`

Compute F1 score for a specific class.

F1 = 2 * (precision * recall) / (precision + recall)

Args:
    predictions: Model predictions (logits) [batch_size, num_classes]
    targets: True labels (sparse) [batch_size]
    num_classes: Total number of classes
    class_idx: Class index to compute F1 score for

Returns:
    Scalar F1 score for the specified class

