# gradient_clipping

## Signature

```python
nabla.nn.gradient_clipping(gradients: list[nabla.core.tensor.Tensor], max_norm: float = 1.0, norm_type: str = 'l2') -> tuple[list[nabla.core.tensor.Tensor], nabla.core.tensor.Tensor]
```

**Source**: `nabla.nn.functional.utils.regularization`

Apply gradient clipping to prevent exploding gradients.

Args:
    gradients: List of gradient tensors
    max_norm: Maximum allowed gradient norm
    norm_type: Type of norm to use ("l2" or "l1")

Returns:
    Tuple of (clipped_gradients, total_norm)

