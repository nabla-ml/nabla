# xavier_normal

## Signature

```python
nabla.nn.xavier_normal(shape: tuple[int, ...], seed: int | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.init.variance_scaling`

Xavier/Glorot normal initialization.

Uses normal distribution with std = sqrt(2/(fan_in + fan_out)) which
is optimal for sigmoid/tanh activations.

Args:
    shape: Shape of the parameter tensor
    seed: Random seed for reproducibility

Returns:
    Initialized parameter tensor

