# init_sgd_state

## Signature

```python
nabla.nn.init_sgd_state(params: list[nabla.core.tensor.Tensor]) -> list[nabla.core.tensor.Tensor]
```

**Source**: `nabla.nn.functional.optim.sgd`

Initialize SGD momentum states.

Args:
    params: List of parameter tensors

Returns:
    List of zero-initialized momentum states

