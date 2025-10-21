# exponential_decay_schedule

## Signature

```python
nabla.nn.exponential_decay_schedule(initial_lr: float = 0.001, decay_factor: float = 0.95, decay_every: int = 1000) -> collections.abc.Callable[[int], float]
```

**Source**: `nabla.nn.functional.optim.schedules`

Exponential decay learning rate schedule.

Args:
    initial_lr: Initial learning rate
    decay_factor: Factor to multiply learning rate by
    decay_every: Apply decay every N epochs

Returns:
    Function that takes epoch and returns learning rate

