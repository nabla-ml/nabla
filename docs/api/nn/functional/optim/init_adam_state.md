# init_adam_state

## Signature

```python
nabla.nn.init_adam_state(params: list[nabla.core.tensor.Tensor]) -> tuple[list[nabla.core.tensor.Tensor], list[nabla.core.tensor.Tensor]]
```

**Source**: `nabla.nn.functional.optim.adam`

Initialize Adam optimizer states.

Args:
    params: List of parameter tensors

Returns:
    Tuple of (m_states, v_states) - zero-initialized moment estimates

