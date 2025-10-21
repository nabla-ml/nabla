# swish

## Signature

```python
nabla.nn.swish(x: nabla.core.tensor.Tensor, beta: float = 1.0) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.activations`

Swish (SiLU) activation function.

Swish(x) = x * sigmoid(β * x)
When β = 1, this is SiLU (Sigmoid Linear Unit).

Args:
    x: Input tensor
    beta: Scaling factor for sigmoid

Returns:
    Tensor with Swish applied element-wise

