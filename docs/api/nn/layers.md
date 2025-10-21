# Layers

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

(128, 30)
```python
>>> import nabla as nb
>>> from nabla.nn import Linear
>>> layer = Linear(20, 30)
>>> input = nb.rand((128, 20))
>>> output = layer(input)
>>> print(output.shape)
```


---
