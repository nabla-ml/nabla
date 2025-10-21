# cosine_annealing_schedule

## Signature

```python
nabla.nn.cosine_annealing_schedule(initial_lr: float = 0.001, min_lr: float = 1e-06, period: int = 1000) -> collections.abc.Callable[[int], float]
```

**Source**: `nabla.nn.functional.optim.schedules`

Cosine annealing learning rate schedule.

Args:
    initial_lr: Initial learning rate
    min_lr: Minimum learning rate
    period: Number of epochs for one complete cosine cycle

Returns:
    Function that takes epoch and returns learning rate

