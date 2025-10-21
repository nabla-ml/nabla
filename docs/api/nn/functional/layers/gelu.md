# gelu

## Signature

```python
nabla.nn.gelu(x: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.activations`

Gaussian Error Linear Unit activation function.

GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution.
Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

Args:
    x: Input tensor

Returns:
    Tensor with GELU applied element-wise

