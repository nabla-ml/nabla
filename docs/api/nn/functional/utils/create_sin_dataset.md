# create_sin_dataset

## Signature

```python
nabla.nn.create_sin_dataset(batch_size: int = 256, sin_periods: int = 8) -> tuple[nabla.core.tensor.Tensor, nabla.core.tensor.Tensor]
```

**Source**: `nabla.nn.functional.utils.training`

Create the 8-Period sin dataset from mlp_train_jit.py.

Args:
    batch_size: Number of samples to generate
    sin_periods: Number of sin periods in [0, 1] interval

Returns:
    Tuple of (x, targets) where targets are sin function values

