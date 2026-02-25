# Functional API

## `linear`

```python
def linear(x: 'Tensor', weight: 'Tensor', bias: 'Tensor | None' = None) -> 'Tensor':
```
Apply a linear projection: y = x @ weight + bias.


---
## `layer_norm`

```python
def layer_norm(x: 'Tensor', weight: 'Tensor | None' = None, bias: 'Tensor | None' = None, eps: 'float' = 1e-05, axis: 'int | tuple[int, ...]' = -1) -> 'Tensor':
```
Apply layer normalization over one or more axes.


---
## `dropout`

```python
def dropout(x: 'Tensor', p: 'float' = 0.5, training: 'bool' = True) -> 'Tensor':
```
Apply dropout: randomly zero elements with probability *p*.

During evaluation (``training=False``) the input is returned unchanged.
Uses inverted-dropout scaling so no adjustment is needed at test time.


---
## `embedding`

```python
def embedding(indices: 'Tensor', weight: 'Tensor') -> 'Tensor':
```
Look up rows of *weight* by integer *indices*.

**Parameters**

- **`indices`** : `Tensor` – Integer tensor of arbitrary shape ``(*)``.
- **`weight`** : `Tensor` – Embedding matrix of shape ``(num_embeddings, embedding_dim)``.

**Returns**

**`Tensor of shape ``(*, embedding_dim)``.`** – 


---
## `scaled_dot_product_attention`

```python
def scaled_dot_product_attention(query: 'Tensor', key: 'Tensor', value: 'Tensor', attn_mask: 'Tensor | None' = None, dropout_p: 'float' = 0.0, is_causal: 'bool' = False, training: 'bool' = True) -> 'Tensor':
```
Scaled dot-product attention (functional).

**Parameters**

- **`query`** : `Tensor  ``(..., seq_q, d_k)``` – None
- **`key`** : `Tensor  ``(..., seq_k, d_k)``` – None
- **`value`** : `Tensor  ``(..., seq_k, d_v)``` – None
- **`attn_mask`** : `optional additive mask broadcastable to ``(..., seq_q, seq_k)``` – None
- **`dropout_p`** : `dropout probability on attention weights (training only)` – None
- **`is_causal`** : `if True, apply a causal (lower-triangular) mask` – None
- **`training`** : `whether we are in training mode (affects dropout)` – None

**Returns**

**`Tensor of shape ``(..., seq_q, d_v)```** – 


---
## `mse_loss`

```python
def mse_loss(predictions: 'Tensor', targets: 'Tensor') -> 'Tensor':
```
Mean squared error loss.


---
## `cross_entropy_loss`

```python
def cross_entropy_loss(logits: 'Tensor', targets: 'Tensor', axis: 'int' = -1) -> 'Tensor':
```
Cross-entropy loss.

**Parameters**

- **`logits`** : `Tensor` – Unnormalized predictions of shape ``(batch, ..., num_classes)``.
- **`targets`** : `Tensor` – Either one-hot labels with the same shape as *logits*, or
integer class indices of shape ``(batch, ...)``.  When the rank of
*targets* is one less than *logits* the targets are treated as
class indices and converted to one-hot internally.
- **`axis`** : `int` – The class axis for softmax (default ``-1``).


---
## `xavier_normal`

```python
def xavier_normal(shape: 'tuple[int, ...]', *, dtype: 'DType' = float32, device: 'str | None' = None) -> 'Tensor':
```
Xavier/Glorot normal initializer for dense layers.


---
## `he_normal`

```python
def he_normal(shape: 'tuple[int, ...]', *, dtype: 'DType' = float32, device: 'str | None' = None) -> 'Tensor':
```
He normal initializer.


---
