# dropout

## Signature

```python
nabla.nn.dropout(x: nabla.core.tensor.Tensor, p: float = 0.5, training: bool = True, seed: int | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.utils.regularization`

Apply dropout regularization.

During training, randomly sets elements to zero with probability p.
During inference, scales all elements by (1-p) to maintain expected values.

Args:
    x: Input tensor
    p: Dropout probability (fraction of elements to set to zero)
    training: Whether in training mode (apply dropout) or inference mode
    seed: Random seed for reproducibility

Returns:
    Tensor with dropout applied

