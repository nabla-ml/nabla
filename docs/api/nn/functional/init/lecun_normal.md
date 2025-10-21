# lecun_normal

## Signature

```python
nabla.nn.lecun_normal(shape: tuple[int, ...], seed: int | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.init.variance_scaling`

LeCun normal initialization.

Uses normal distribution with std = sqrt(1/fan_in) which is optimal
for SELU activations.

Args:
    shape: Shape of the parameter tensor
    seed: Random seed for reproducibility

Returns:
    Initialized parameter tensor

