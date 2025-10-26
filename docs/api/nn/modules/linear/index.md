# Linear Layers

## `Linear`

```python
class Linear(in_features: 'int', out_features: 'int', bias: 'bool' = True):
```
Applies a linear transformation to the incoming data: y = xA^T + b.


### Methods

#### `extra_repr`
```python
def extra_repr(self) -> 'str':
```

#### `forward`
```python
def forward(self, x: 'nb.Tensor') -> 'nb.Tensor':
```

---
