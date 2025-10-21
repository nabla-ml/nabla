# recall

## Signature

```python
nabla.nn.recall(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor, num_classes: int, class_idx: int = 0) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.metrics`

Compute recall for a specific class.

Recall = TP / (TP + FN)

Args:
    predictions: Model predictions (logits) [batch_size, num_classes]
    targets: True labels (sparse) [batch_size]
    num_classes: Total number of classes
    class_idx: Class index to compute recall for

Returns:
    Scalar recall value for the specified class

