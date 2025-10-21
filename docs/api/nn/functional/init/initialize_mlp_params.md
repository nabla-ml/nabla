# initialize_mlp_params

## Signature

```python
nabla.nn.initialize_mlp_params(layers: list[int], seed: int = 42) -> list[nabla.core.tensor.Tensor]
```

**Source**: `nabla.nn.functional.init.variance_scaling`

Initialize MLP parameters with specialized strategy for complex functions.

This is the original initialization strategy from mlp_train_jit.py,
optimized for learning high-frequency functions.

Args:
    layers: List of layer sizes [input, hidden1, hidden2, ..., output]
    seed: Random seed for reproducibility

Returns:
    List of parameter tensors [W1, b1, W2, b2, ...]

