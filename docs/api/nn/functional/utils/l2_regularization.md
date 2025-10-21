# l2_regularization

## Signature

```python
nabla.nn.l2_regularization(params: list[nabla.core.tensor.Tensor], weight: float = 0.01) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.regularization`

Compute L2 (Ridge) regularization loss.

L2 regularization adds a penalty equal to the sum of squares of parameters.
This encourages small parameter values and helps prevent overfitting.

Args:
    params: List of parameter tensors (typically weights)
    weight: Regularization strength

Returns:
    Scalar L2 regularization loss

