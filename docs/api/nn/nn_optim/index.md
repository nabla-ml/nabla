# Optimizers (nabla.nn.optim)

## `Optimizer`

```python
class Optimizer(params: 'Any') -> 'None':
```
Base class for stateful optimizers backed by pure functional steps.


### Methods

#### `step`
```python
def step(self, grads: 'Any') -> 'Any':
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
def step(self, grads: 'Any') -> 'Any':
```

---
## `AdamW`

```python
class AdamW(params: 'Any', *, lr: 'float', betas: 'tuple[float, float]' = (0.9, 0.999), eps: 'float' = 1e-08, weight_decay: 'float' = 0.0) -> 'None':
```
Base class for stateful optimizers backed by pure functional steps.


### Methods

#### `step`
```python
def step(self, grads: 'Any') -> 'Any':
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
def adamw_step(param: 'Tensor', grad: 'Tensor', m: 'Tensor', v: 'Tensor', step: 'int', *, lr: 'float', beta1: 'float' = 0.9, beta2: 'float' = 0.999, eps: 'float' = 1e-08, weight_decay: 'float' = 0.0) -> 'tuple[Tensor, Tensor, Tensor]':
```
Single-tensor AdamW update.


---
