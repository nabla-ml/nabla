# elastic_net_regularization

## Signature

```python
nabla.nn.elastic_net_regularization(params: list[nabla.core.tensor.Tensor], l1_weight: float = 0.01, l2_weight: float = 0.01, l1_ratio: float = 0.5) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.regularization`

Compute Elastic Net regularization loss.

Elastic Net combines L1 and L2 regularization:
ElasticNet = l1_ratio * L1 + (1 - l1_ratio) * L2

Args:
    params: List of parameter tensors (typically weights)
    l1_weight: L1 regularization strength
    l2_weight: L2 regularization strength
    l1_ratio: Ratio of L1 to L2 regularization (0 = pure L2, 1 = pure L1)

Returns:
    Scalar Elastic Net regularization loss

