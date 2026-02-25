# Modules

## `Module`

```python
class Module() -> 'None':
```
Base class for all neural-network modules.

Subclasses must override :meth:`forward`. Parameters (tensors with
``requires_grad=True``) assigned to attributes are automatically
tracked and yielded by :meth:`parameters`. Submodules assigned to
attributes are recursively tracked by :meth:`modules`.

Modules are registered as PyTree nodes, so they can be passed
directly to transforms like :func:`~nabla.vmap`, :func:`~nabla.grad`,
and :func:`~nabla.compile` without any special wrapping.

Example::

    class MLP(nabla.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc1 = nabla.nn.Linear(in_dim, 64)
            self.fc2 = nabla.nn.Linear(64, out_dim)

        def forward(self, x):
            return self.fc2(nabla.relu(self.fc1(x)))


### Methods

#### `backward`
```python
def backward(self, loss: 'Tensor', gradient: 'Tensor | None' = None, retain_graph: 'bool' = False, create_graph: 'bool' = False, *, realize_grads: 'bool | None' = None) -> 'None':
```
PyTorch-style backward convenience attached to Module.

Optionally realizes all parameter gradients after backward.


#### `buffers`
```python
def buffers(self) -> 'Iterator[Tensor]':
```

#### `eval`
```python
def eval(self) -> 'Module':
```

#### `extra_repr`
```python
def extra_repr(self) -> 'str':
```

#### `forward`
```python
def forward(self, *args: 'Any', **kwargs: 'Any') -> 'Any':
```

#### `load_state_dict`
```python
def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]') -> 'None':
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
def register_buffer(self, name: 'str', tensor: 'Tensor | None') -> 'None':
```

#### `state_dict`
```python
def state_dict(self) -> 'OrderedDict[str, Tensor]':
```

#### `train`
```python
def train(self) -> 'Module':
```

#### `zero_grad`
```python
def zero_grad(self) -> 'None':
```

---
## `Linear`

```python
class Linear(in_features: 'int', out_features: 'int', bias: 'bool' = True, *, dtype: 'DType' = float32) -> 'None':
```
Apply a linear transformation: ``y = x @ W.T + b``.

**Parameters**

- **`in_features`** – Number of input features.
- **`out_features`** – Number of output features.
- **`bias`** – If ``True`` (default), adds a learnable bias term.
- **`dtype`** – Dtype for weight and bias. Default: ``float32``.


### Methods

#### `extra_repr`
```python
def extra_repr(self) -> 'str':
```

#### `forward`
```python
def forward(self, x: 'Tensor') -> 'Tensor':
```

---
## `LayerNorm`

```python
class LayerNorm(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True, *, dtype: 'DType' = float32) -> 'None':
```
Apply layer normalization over the last ``len(normalized_shape)`` dimensions.

Normalises inputs as ``(x - mean) / sqrt(var + eps)`` and then applies
a learnable per-element affine transform when *elementwise_affine* is
``True``.

**Parameters**

- **`normalized_shape`** – Input shape from an expected input of size
``(*, normalized_shape[0], ..., normalized_shape[-1])``.
Can be an ``int`` for the common last-dimension case.
- **`eps`** – Value added to the denominator for numerical stability.
Default: ``1e-5``.
- **`elementwise_affine`** – If ``True`` (default), learnable ``weight``
(initialized to 1) and ``bias`` (initialized to 0) are added.
- **`dtype`** – Dtype for weight and bias. Default: ``float32``.


### Methods

#### `extra_repr`
```python
def extra_repr(self) -> 'str':
```

#### `forward`
```python
def forward(self, x: 'Tensor') -> 'Tensor':
```

---
## `Dropout`

```python
class Dropout(p: 'float' = 0.5) -> 'None':
```
Randomly zero elements of the input with probability *p* (Bernoulli dropout).

Elements that are not zeroed are scaled by ``1 / (1 - p)`` (inverted dropout)
so that the expected value of each element is unchanged. Set to ``eval()``
mode to disable dropout during inference.

**Parameters**

- **`p`** – Probability of an element being zeroed. Default: ``0.5``.


---
## `Embedding`

```python
class Embedding(num_embeddings: 'int', embedding_dim: 'int', *, dtype: 'DType' = float32) -> 'None':
```
A learnable lookup table mapping integer indices to dense vectors.

**Parameters**

- **`num_embeddings`** : `int` – Size of the vocabulary (number of rows).
- **`embedding_dim`** : `int` – Dimensionality of each embedding vector.


### Methods

#### `extra_repr`
```python
def extra_repr(self) -> 'str':
```

#### `forward`
```python
def forward(self, indices: 'Tensor') -> 'Tensor':
```

---
## `MultiHeadAttention`

```python
class MultiHeadAttention(d_model: 'int', num_heads: 'int', dropout: 'float' = 0.0, bias: 'bool' = True, *, dtype: 'DType' = float32) -> 'None':
```
Multi-head attention as described in *Attention Is All You Need*.

**Parameters**

- **`d_model`** : `int` – Total model dimensionality.
- **`num_heads`** : `int` – Number of parallel attention heads.  ``d_model`` must be divisible
by ``num_heads``.
- **`dropout`** : `float` – Dropout probability on attention weights (applied during training).
- **`bias`** : `bool` – Whether the linear projections include bias terms.


### Methods

#### `extra_repr`
```python
def extra_repr(self) -> 'str':
```

#### `forward`
```python
def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', attn_mask: 'Tensor | None' = None, is_causal: 'bool' = False) -> 'Tensor':
```
Run multi-head attention.

**Parameters**

- **`query, key, value`** : `Tensor ``(batch, seq_*, d_model)``` – None
- **`attn_mask`** : `optional additive mask ``(..., seq_q, seq_k)``` – None
- **`is_causal`** : `apply a causal mask` – None

**Returns**

**`Tensor ``(batch, seq_q, d_model)```** – 


---
## `TransformerEncoderLayer`

```python
class TransformerEncoderLayer(d_model: 'int', num_heads: 'int', dim_feedforward: 'int' = 2048, dropout: 'float' = 0.1, *, dtype: 'DType' = float32) -> 'None':
```
A single Transformer encoder layer (pre-norm variant).

Structure::

    x ─→ LayerNorm ─→ MultiHeadAttention ─→ Dropout ─→ + ─→
    │                                                    ↑
    └────────────────────────────────────────────────────┘
    x ─→ LayerNorm ─→ FFN ─→ Dropout ─→ + ─→
    │                                      ↑
    └──────────────────────────────────────┘

**Parameters**

- **`d_model`** : `int` – Model dimensionality.
- **`num_heads`** : `int` – Number of attention heads.
- **`dim_feedforward`** : `int` – Hidden size of the position-wise feed-forward network.
- **`dropout`** : `float` – Dropout probability.


### Methods

#### `forward`
```python
def forward(self, src: 'Tensor', src_mask: 'Tensor | None' = None, is_causal: 'bool' = False) -> 'Tensor':
```

---
## `TransformerDecoderLayer`

```python
class TransformerDecoderLayer(d_model: 'int', num_heads: 'int', dim_feedforward: 'int' = 2048, dropout: 'float' = 0.1, *, dtype: 'DType' = float32) -> 'None':
```
A single Transformer decoder layer (pre-norm variant).

Structure::

    tgt ─→ LayerNorm ─→ Masked-Self-Attention ─→ Dropout ─→ + ─→
    tgt ─→ LayerNorm ─→ Cross-Attention(tgt, memory) ─→ Dropout ─→ + ─→
    tgt ─→ LayerNorm ─→ FFN ─→ Dropout ─→ + ─→

**Parameters**

- **`d_model, num_heads, dim_feedforward, dropout`** : `same as encoder layer.` – None


### Methods

#### `forward`
```python
def forward(self, tgt: 'Tensor', memory: 'Tensor', tgt_mask: 'Tensor | None' = None, memory_mask: 'Tensor | None' = None, is_causal: 'bool' = False) -> 'Tensor':
```

---
## `Sequential`

```python
class Sequential(*args: 'Any') -> 'None':
```
Chain modules sequentially — output of each becomes input to the next.

**Parameters**

- **`*args`** – Either a flat sequence of :class:`Module` instances, or a
single :class:`OrderedDict` mapping string names to modules.


---
## `ReLU`

```python
class ReLU() -> 'None':
```
Apply the Rectified Linear Unit activation: ``max(0, x)``.


---
## `GELU`

```python
class GELU() -> 'None':
```
Apply the Gaussian Error Linear Unit activation.


---
## `Sigmoid`

```python
class Sigmoid() -> 'None':
```
Apply the sigmoid activation: ``1 / (1 + exp(-x))``.


---
## `Tanh`

```python
class Tanh() -> 'None':
```
Apply the hyperbolic tangent activation: ``tanh(x)``.


---
## `SiLU`

```python
class SiLU() -> 'None':
```
Apply the Sigmoid Linear Unit (Swish) activation: ``x * sigmoid(x)``.


---
## `Softmax`

```python
class Softmax(axis: 'int' = -1) -> 'None':
```
Apply softmax along a given axis (default: last).


---
