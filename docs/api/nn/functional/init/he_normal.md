# he_normal

## Signature

```python
nabla.nn.he_normal(shape: tuple[int, ...], seed: int | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.init.variance_scaling`

He normal initialization for ReLU networks.

Uses normal distribution with std = sqrt(2/fan_in) which is optimal
for ReLU activations.

Args:
    shape: Shape of the parameter tensor
    seed: Random seed for reproducibility

Returns:
    Initialized parameter tensor

