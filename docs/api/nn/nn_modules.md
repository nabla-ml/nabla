# Modules (Stateful OOP Layers)

## `Linear`

```python
class Linear(in_features: 'int', out_features: 'int', bias: 'bool' = True):
```
Applies a linear transformation: y = x @ W + b

**Parameters**

- **`in_features`** : `int` – Size of input features
- **`out_features`** : `int` – Size of output features
- **`bias`** : `bool`, optional – If True, adds a learnable bias (default: True)
- **`weight`** : `Tensor` – Learnable weights of shape (in_features, out_features)
- **`bias`** : `Tensor` – Learnable bias of shape (1, out_features) if bias=True

**Examples**

--
```python
>>> import nabla as nb
>>> from nabla.nn import Linear
>>> layer = Linear(20, 30)
>>> input = nb.rand((128, 20))
>>> output = layer(input)
>>> print(output.shape)
(128, 30)
```

---
## `Sequential`

```python
class Sequential(*modules):
```
Sequential container that applies modules in order.

Modules will be added to the container in the order they are passed
in the constructor. The forward() method automatically chains them.

**Parameters**

- **`*modules`** : `Module` – Variable number of modules to add sequentially

**Examples**

--
```python
>>> from nabla.nn import Sequential, Linear
>>> model = Sequential(
...     Linear(10, 20),
...     Linear(20, 10)
... )
>>> output = model(input)  # Automatically applies layers in order
```

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

**Parameters**

- **`*modules`** : `Module` – Variable number of modules to add to the list

**Examples**

--
```python
>>> from nabla.nn import ModuleList, Linear
>>> layers = ModuleList(
...     Linear(10, 20),
...     Linear(20, 10)
... )
>>> for layer in layers:
...     x = layer(x)
```

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

**Parameters**

- **`modules`** – Optional dict of modules to initialize with


---
