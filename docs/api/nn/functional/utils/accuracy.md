# accuracy

## Signature

```python
nabla.nn.accuracy(predictions: nabla.core.tensor.Tensor, targets: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.metrics`

Compute classification accuracy.

Args:
    predictions: Model predictions - either logits/probabilities [batch_size, num_classes]
                or class indices [batch_size]
    targets: True labels - either one-hot [batch_size, num_classes] or indices [batch_size]

Returns:
    Scalar accuracy value between 0 and 1

