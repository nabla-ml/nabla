# Containers

## `Sequential`

```python
class Sequential(*args):
```
A sequential container.

Modules will be added to it in the order they are passed in the constructor.


### Methods

#### `forward`
```python
def forward(self, x):
```

---
## `ModuleList`

```python
class ModuleList(modules: 'list[Module] | None' = None):
```
Holds submodules in a list.

ModuleList can be indexed like a regular Python list, but modules it
contains are properly registered, and will be visible by all Module methods.


### Methods

#### `append`
```python
def append(self, module: 'Module') -> 'None':
```

#### `forward`
```python
def forward(self, *args, **kwargs):
```

---
## `ModuleDict`

```python
class ModuleDict(modules: 'dict[str, Module] | None' = None):
```
Holds submodules in a dictionary.

ModuleDict can be indexed like a regular Python dictionary, but modules it
contains are properly registered, and will be visible by all Module methods.


### Methods

#### `forward`
```python
def forward(self, *args, **kwargs):
```

#### `items`
```python
def items(self):
```

#### `keys`
```python
def keys(self):
```

#### `values`
```python
def values(self):
```

---
