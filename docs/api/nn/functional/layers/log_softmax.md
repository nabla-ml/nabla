# log_softmax

## Signature

```python
nabla.nn.log_softmax(x: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.nn.functional.layers.activations`

Log-softmax activation function.

Args:
    x: Input tensor
    axis: Axis along which to compute log-softmax

Returns:
    Tensor with log-softmax applied along specified axis

