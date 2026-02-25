# Optim

## `Optimizer`

```python
class Optimizer(params: 'Any') -> 'None':
```
Base class for stateful optimizers backed by pure functional steps.


### Methods

#### `step`
```python
def step(self, grads: 'Any' = None) -> 'Any':
```

---
## `SGD`

```python
class SGD(params: 'Any', *, lr: 'float', momentum: 'float' = 0.0, weight_decay: 'float' = 0.0) -> 'None':
```
Stateful SGD optimizer with optional momentum and weight decay.

Usage::

    optimizer = SGD(params, lr=0.01, momentum=0.9)
    new_params = optimizer.step(grads)


### Methods

#### `step`
```python
def step(self, grads: 'Any' = None) -> 'Any':
```

---
## `AdamW`

```python
class AdamW(params: 'Any', *, lr: 'float', betas: 'tuple[float, float]' = (0.9, 0.999), eps: 'float' = 1e-08, weight_decay: 'float' = 0.0) -> 'None':
```
Stateful AdamW optimizer with decoupled weight decay.

Implements the Adam algorithm with decoupled weight decay regularisation
from Loshchilov & Hutter (2019).

**Parameters**

- **`params`** – Model parameters (a tensor or pytree of tensors).
- **`lr`** – Learning rate.
- **`betas`** – Coefficients for computing running averages of gradient
and its square. Default: ``(0.9, 0.999)``.
- **`eps`** – Small constant for numerical stability. Default: ``1e-8``.
- **`weight_decay`** – Decoupled weight decay coefficient. Default: ``0.0``.


### Methods

#### `step`
```python
def step(self, grads: 'Any' = None) -> 'Any':
```

---
## `sgd_step`

```python
def sgd_step(param: 'Tensor', grad: 'Tensor', momentum_buffer: 'Tensor | None' = None, *, lr: 'float', weight_decay: 'float' = 0.0, momentum: 'float' = 0.0) -> 'tuple[Tensor, Tensor | None]':
```
Single-tensor SGD update.

Returns ``(new_param, new_momentum_buffer)``.


---
## `adamw_step`

```python
def adamw_step(param: 'Tensor', grad: 'Tensor', m: 'Tensor', v: 'Tensor', step: 'int | float | Tensor', *, lr: 'float', beta1: 'float' = 0.9, beta2: 'float' = 0.999, eps: 'float' = 1e-08, weight_decay: 'float' = 0.0, bias_correction: 'bool' = True) -> 'tuple[Tensor, Tensor, Tensor]':
```
Single-tensor AdamW update.

Handles both scalar and tensor ``step`` (the latter is needed inside
``@nb.compile`` where the step counter lives as a 0-D tensor).


---
## `sgd_update`

```python
def sgd_update(params: 'Any', grads: 'Any', state: 'dict[str, Any] | None' = None, *, lr: 'float', momentum: 'float' = 0.0, weight_decay: 'float' = 0.0) -> 'tuple[Any, dict[str, Any]]':
```
Functional SGD update on pytrees (mirrors ``adamw_update``).

**Parameters**

- **`params`** : `pytree` – Current model parameters.
- **`grads`** : `pytree` – Gradients matching the *params* structure.
- **`state`** : `dict`, optional – Optimizer state containing ``"momentum_buffers"`` and ``"step"``.
If *None* a fresh state is created.
- **`lr, momentum, weight_decay`** : `float` – Standard SGD hyper-parameters.

**Returns**

`tuple` – Updated parameters and optimizer state, with tensors realized
according to the global ``Optimizer`` execution policy.


---
## `adamw_init`

```python
def adamw_init(params: 'Any') -> 'dict[str, Any]':
```
Functional AdamW state init for pytree params.


---
## `adamw_update`

```python
def adamw_update(params: 'Any', grads: 'Any', state: 'dict[str, Any]', *, lr: 'float', weight_decay: 'float' = 0.0, beta1: 'float' = 0.9, beta2: 'float' = 0.999, eps: 'float' = 1e-08, bias_correction: 'bool' = True, realize: 'bool | None' = None) -> 'tuple[Any, dict[str, Any]]':
```
Functional AdamW update on pytrees.

Delegates per-leaf math to :func:`adamw_step` so the update logic
lives in one place.  Handles both scalar and tensor ``step`` (the
latter is produced by ``_normalize_optimizer_state_for_compile``
inside ``@nb.compile``).


---
