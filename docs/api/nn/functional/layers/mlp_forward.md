# mlp_forward

## Signature

```python
nabla.nn.mlp_forward(x: nabla.core.tensor.Tensor, params: list[nabla.core.tensor.Tensor]) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.linear`

MLP forward pass through all layers.

This is the original MLP forward function from mlp_train_jit.py.
Applies ReLU activation to all layers except the last.

Args:
    x: Input tensor of shape (batch_size, input_dim)
    params: List of parameters [W1, b1, W2, b2, ..., Wn, bn]

Returns:
    Output tensor of shape (batch_size, output_dim)

