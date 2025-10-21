# mlp_forward_with_activations

## Signature

```python
nabla.nn.mlp_forward_with_activations(x: nabla.core.tensor.Tensor, params: list[nabla.core.tensor.Tensor], activation: str = 'relu', final_activation: str | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.linear`

MLP forward pass with configurable activations.

Args:
    x: Input tensor of shape (batch_size, input_dim)
    params: List of parameters [W1, b1, W2, b2, ..., Wn, bn]
    activation: Activation function for hidden layers ("relu", "tanh", "sigmoid")
    final_activation: Optional activation for final layer

Returns:
    Output tensor of shape (batch_size, output_dim)

