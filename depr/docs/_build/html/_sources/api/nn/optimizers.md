# Optimizers

## `Optimizer`

```python
class Optimizer(params: 'Iterator[Tensor] | list[Tensor]'):
```
Base class for all optimizers.

Handles parameter updates after gradients are computed via backward().
All optimizer implementations should inherit from this class.

Parameters
----------
params : Iterator[Tensor] or list[Tensor]
    Iterator or list of parameters to optimize

Examples
--------
>>> from nabla.nn import SGD
>>> optimizer = SGD(model.parameters(), lr=0.01)
>>> loss.backward()
>>> optimizer.step()  # Updates parameters
>>> optimizer.zero_grad()  # Clears gradients

---
## `SGD`

```python
class SGD(params: 'Iterator[Tensor] | list[Tensor]', lr: 'float' = 0.01, momentum: 'float' = 0.0, weight_decay: 'float' = 0.0):
```
Stochastic Gradient Descent optimizer.

Implements SGD with optional momentum and weight decay (L2 regularization).

Parameters
----------
params : Iterator[Tensor] or list[Tensor]
    Parameters to optimize
lr : float, optional
    Learning rate (default: 0.01)
momentum : float, optional
    Momentum factor (default: 0.0, no momentum)
weight_decay : float, optional
    Weight decay (L2 penalty) (default: 0.0, no decay)

Examples
--------
>>> from nabla.nn import SGD
>>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
>>> for data, target in dataloader:
...     optimizer.zero_grad()
...     output = model(data)
...     loss = criterion(output, target)
...     loss.backward()
...     optimizer.step()

---
## `Adam`

```python
class Adam(params: 'Iterator[Tensor] | list[Tensor]', lr: 'float' = 0.001, betas: 'tuple[float, float]' = (0.9, 0.999), eps: 'float' = 1e-08, weight_decay: 'float' = 0.0):
```
Adam optimizer (Adaptive Moment Estimation).

Implements Adam algorithm with bias correction. Maintains moving averages
of gradients and their squares for adaptive learning rates.

Args:
    params: Parameters to optimize
    lr: Learning rate (default: 0.001)
    betas: Coefficients for computing running averages of gradient
           and its square (default: (0.9, 0.999))
    eps: Term added to denominator for numerical stability (default: 1e-8)
    weight_decay: Weight decay (L2 penalty) (default: 0.0, no decay)

Example:
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>> 
    >>> for data, target in dataloader:
    ...     optimizer.zero_grad()
    ...     output = model(data)
    ...     loss = criterion(output, target)
    ...     loss.backward()
    ...     optimizer.step()

References:
    Adam: A Method for Stochastic Optimization
    Kingma & Ba, ICLR 2015
    https://arxiv.org/abs/1412.6980

---
