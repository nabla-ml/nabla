# init_adamw_state

## Signature

```python
nabla.nn.init_adamw_state(params: list[nabla.core.tensor.Tensor]) -> tuple[list[nabla.core.tensor.Tensor], list[nabla.core.tensor.Tensor]]
```

**Source**: `nabla.nn.functional.optim.adamw`

Initialize AdamW optimizer state.

Args:
    params: List of parameter tensors

Returns:
    Tuple of (m_states, v_states) - first and second moment estimates

