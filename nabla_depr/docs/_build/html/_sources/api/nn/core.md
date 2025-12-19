# Core

## `Module`

```python
class Module():
```
Base class for all neural network modules (PyTorch-like nn.Module).

Your models should subclass this class and implement the forward() method.

Automatically tracks:
- Parameters (Tensors with requires_grad=True)
- Submodules (nested Module instances)

Provides:
- Recursive parameter access via .parameters()
- Named parameter iteration via .named_parameters()
- Module tree iteration via .modules()
- Gradient zeroing via .zero_grad()
- Callable interface: model(x) calls model.forward(x)

Examples
--------
>>> from nabla.nn import Module, Linear
>>> class MLP(Module):
...     def __init__(self, layer_sizes):
...         super().__init__()
...         self.layers = [Linear(layer_sizes[i], layer_sizes[i+1])
...                       for i in range(len(layer_sizes)-1)]
...     def forward(self, x):
...         for layer in self.layers:
...             x = layer(x)
...         return x
>>> model = MLP([10, 20, 10])
>>> params = list(model.parameters())  # Gets all params recursively

---
