# Core

## `Optimizer`

```python
class Optimizer(params, defaults: 'dict'):
```
Base class for all optimizers, inspired by PyTorch's optim.Optimizer.

**Parameters**

- **`params`** : `iterable` – An iterable of parameters to optimize or dicts defining
parameter groups.
- **`defaults`** : `dict` – A dict containing default values of optimization
options (e.g. learning rate, momentum).


---
