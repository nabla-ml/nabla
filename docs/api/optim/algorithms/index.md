# Algorithms

## `SGD`

```python
class SGD(params, lr: float, momentum: float = 0, weight_decay: float = 0):
```
Implements stochastic gradient descent (optionally with momentum).


### Methods

#### `step`
```python
def step(self) -> None:
```
Performs a single optimization step.


---
## `Adam`

```python
class Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
```
Implements Adam algorithm.


### Methods

#### `step`
```python
def step(self) -> None:
```
Performs a single optimization step.


---
