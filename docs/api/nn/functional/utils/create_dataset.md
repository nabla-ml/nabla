# create_dataset

## Signature

```python
nabla.nn.create_dataset(batch_size: int, input_dim: int, seed: int | None = None) -> tuple[nabla.core.tensor.Tensor, nabla.core.tensor.Tensor]
```

**Source**: `nabla.nn.functional.utils.training`

Create a simple random dataset for testing.

Args:
    batch_size: Number of samples
    input_dim: Input dimension
    seed: Random seed for reproducibility

Returns:
    Tuple of (inputs, targets)

