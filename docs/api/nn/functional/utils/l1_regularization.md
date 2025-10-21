# l1_regularization

## Signature

```python
nabla.nn.l1_regularization(params: list[nabla.core.tensor.Tensor], weight: float = 0.01) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.regularization`

Compute L1 (Lasso) regularization loss.

L1 regularization adds a penalty equal to the sum of absolute values of parameters.
This encourages sparsity in the model parameters.

Args:
    params: List of parameter tensors (typically weights)
    weight: Regularization strength

Returns:
    Scalar L1 regularization loss

