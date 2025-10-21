# softmax

## Signature

```python
nabla.nn.softmax(x: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.activations`

Softmax activation function.

Args:
    x: Input tensor
    axis: Axis along which to compute softmax

Returns:
    Tensor with softmax applied along specified axis

