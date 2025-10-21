# leaky_relu

## Signature

```python
nabla.nn.leaky_relu(x: nabla.core.tensor.Tensor, negative_slope: float = 0.01) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.activations`

Leaky ReLU activation function.

Args:
    x: Input tensor
    negative_slope: Slope for negative values

Returns:
    Tensor with Leaky ReLU applied element-wise

