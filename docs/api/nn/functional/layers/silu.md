# silu

## Signature

```python
nabla.nn.silu(x: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.activations`

Sigmoid Linear Unit (SiLU) activation function.

SiLU(x) = x * sigmoid(x) = Swish(x, β=1)

Args:
    x: Input tensor

Returns:
    Tensor with SiLU applied element-wise

