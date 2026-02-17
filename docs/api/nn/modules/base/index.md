# Base Class

## `Module`

```python
class Module():
```
Base class for all neural network modules, inspired by PyTorch's nn.Module.


### Methods

#### `buffers`
```python
def buffers(self) -> 'Iterator[Tensor]':
```

#### `eval`
```python
def eval(self):
```

#### `extra_repr`
```python
def extra_repr(self) -> 'str':
```

#### `forward`
```python
def forward(self, *args, **kwargs):
```

#### `load_state_dict`
```python
def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]'):
```

#### `modules`
```python
def modules(self) -> 'Iterator[Module]':
```

#### `named_buffers`
```python
def named_buffers(self, prefix: 'str' = '') -> 'Iterator[tuple[str, Tensor]]':
```

#### `named_parameters`
```python
def named_parameters(self, prefix: 'str' = '') -> 'Iterator[tuple[str, Tensor]]':
```

#### `parameters`
```python
def parameters(self) -> 'Iterator[Tensor]':
```

#### `register_buffer`
```python
def register_buffer(self, name: 'str', tensor: 'Tensor | None'):
```
Adds a persistent buffer to the module.


#### `state_dict`
```python
def state_dict(self) -> 'OrderedDict[str, Tensor]':
```

#### `train`
```python
def train(self):
```

#### `zero_grad`
```python
def zero_grad(self) -> 'None':
```

---
