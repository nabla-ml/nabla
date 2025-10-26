# Containers

## `Sequential`

```python
class Sequential(*args):
```
A sequential container.

Modules will be added to it in the order they are passed in the constructor.


---
## `ModuleList`

```python
class ModuleList(modules: 'list[Module] | None' = None):
```
Holds submodules in a list.

ModuleList can be indexed like a regular Python list, but modules it
contains are properly registered, and will be visible by all Module methods.


---
## `ModuleDict`

```python
class ModuleDict(modules: 'dict[str, Module] | None' = None):
```
Holds submodules in a dictionary.

ModuleDict can be indexed like a regular Python dictionary, but modules it
contains are properly registered, and will be visible by all Module methods.


---
