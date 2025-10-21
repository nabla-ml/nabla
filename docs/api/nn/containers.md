# Containers

## `Sequential`

```python
class Sequential(*modules):
```
Sequential container that applies modules in order.

Modules will be added to the container in the order they are passed
in the constructor. The forward() method automatically chains them.

Parameters
----------
*modules : Module
    Variable number of modules to add sequentially

Examples
--------
>>> from nabla.nn import Sequential, Linear
>>> model = Sequential(
...     Linear(10, 20),
...     Linear(20, 10)
... )
>>> output = model(input)  # Automatically applies layers in order

---
## `ModuleList`

```python
class ModuleList(*modules):
```
Container that holds modules in a list.

Like PyTorch's nn.ModuleList - modules are properly registered and can
be indexed, iterated, and appended to. The modules are registered as
submodules so their parameters are collected.

Note: ModuleList does not define forward() - it's a container that you
use within your own modules.

Parameters
----------
*modules : Module
    Variable number of modules to add to the list

Examples
--------
>>> from nabla.nn import ModuleList, Linear
>>> layers = ModuleList(
...     Linear(10, 20),
...     Linear(20, 10)
... )
>>> for layer in layers:
...     x = layer(x)

---
## `ModuleDict`

```python
class ModuleDict(modules: 'dict[str, Module] | None' = None):
```
Container that holds modules in a dictionary.

Like PyTorch's nn.ModuleDict - modules are properly registered with
string keys. Can be accessed, iterated, and modified like a dict.

Note: ModuleDict does not define forward() - it's a container that you
use within your own modules.

Args:
    modules: Optional dict of modules to initialize with

Example:
    >>> components = ModuleDict({
    ...     'encoder': Linear(10, 5),
    ...     'decoder': Linear(5, 10)
    ... })
    >>> encoded = components['encoder'](x)
    >>> decoded = components['decoder'](encoded)

---
