# warmup_cosine_schedule

## Signature

```python
nabla.nn.warmup_cosine_schedule(initial_lr: float = 0.001, warmup_epochs: int = 100, total_epochs: int = 1000, min_lr: float = 1e-06) -> collections.abc.Callable[[int], float]
```

**Source**: `nabla.nn.functional.optim.schedules`

Warmup followed by cosine annealing schedule.

Args:
    initial_lr: Peak learning rate after warmup
    warmup_epochs: Number of epochs for linear warmup
    total_epochs: Total number of training epochs
    min_lr: Minimum learning rate

Returns:
    Function that takes epoch and returns learning rate

