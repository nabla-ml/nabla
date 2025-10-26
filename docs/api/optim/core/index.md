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


### Methods

#### `add_param_group`
```python
def add_param_group(self, param_group: 'dict'):
```
Add a param group to the Optimizer's param_groups list.


#### `load_state_dict`
```python
def load_state_dict(self, state_dict: 'dict'):
```
Loads the optimizer state.

**Parameters**

- **`state_dict`** : `dict` – optimizer state. Should be an object returned
from a call to :meth:`state_dict`.


#### `state_dict`
```python
def state_dict(self) -> 'dict':
```
Returns the state of the optimizer as a dict.

It contains two entries:
* state - a dict holding current optimization state. Its content
    differs between optimizer classes.
* param_groups - a list containing all parameter groups.


#### `step`
```python
def step(self) -> 'None':
```
Performs a single optimization step.


#### `zero_grad`
```python
def zero_grad(self) -> 'None':
```
Sets the gradients of all optimized Tensors to None.


---
