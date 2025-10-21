# create_mlp_config

## Signature

```python
nabla.nn.create_mlp_config(layers: list[int], activation: str = 'relu', final_activation: str | None = None, init_method: str = 'he_normal', seed: int = 42) -> dict
```

**Source**: `nabla.nn.functional.architectures.mlp`

Create MLP configuration dictionary.

Args:
    layers: List of layer sizes [input, hidden1, hidden2, ..., output]
    activation: Activation function for hidden layers
    final_activation: Optional activation for final layer
    init_method: Weight initialization method
    seed: Random seed for reproducibility

Returns:
    Configuration dictionary with params and forward function

