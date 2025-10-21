# learning_rate_schedule

## Signature

```python
nabla.nn.learning_rate_schedule(epoch: int, initial_lr: float = 0.001, decay_factor: float = 0.95, decay_every: int = 1000) -> float
```

**Source**: `nabla.nn.functional.optim.schedules`

Learning rate schedule for complex function learning.

This is the original function from mlp_train_jit.py for backward compatibility.
Consider using exponential_decay_schedule instead for new code.

Args:
    epoch: Current epoch number
    initial_lr: Initial learning rate
    decay_factor: Factor to multiply learning rate by
    decay_every: Apply decay every N epochs

Returns:
    Learning rate for the current epoch

