# linear_forward

## Signature

```python
nabla.nn.linear_forward(x: nabla.core.tensor.Tensor, weight: nabla.core.tensor.Tensor, bias: nabla.core.tensor.Tensor | None = None) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.linear`

Forward pass through a linear layer.

Computes: output = x @ weight + bias

Args:
    x: Input tensor of shape (batch_size, in_features)
    weight: Weight tensor of shape (in_features, out_features)
    bias: Optional bias tensor of shape (1, out_features) or (out_features,)

Returns:
    Output tensor of shape (batch_size, out_features)

