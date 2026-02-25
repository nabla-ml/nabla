# LoRA

## `init_lora_adapter`

```python
def init_lora_adapter(weight: 'Tensor', rank: 'int', init_std: 'float' = 0.01, dtype: 'DType | None' = None) -> 'dict[str, Tensor]':
```
Initialise LoRA adapter matrices ``A`` and ``B`` for a 2D weight.

Following Hu et al. (2021), ``A`` is initialised with Gaussian noise and
``B`` is zero-initialised so the adapter adds zero at the start of training.

**Parameters**

- **`weight`** – The frozen 2D weight tensor to adapt. Shape ``(in, out)``.
- **`rank`** – Intrinsic rank of the low-rank decomposition. Must be > 0.
- **`init_std`** – Standard deviation for initialising ``A``. Default: ``0.01``.
- **`dtype`**, default: `*weight*'s dtype` – Optional dtype override. Defaults to *weight*'s dtype.

**Returns**

**```{'A'`** – Tensor(in, rank), 'B': Tensor(rank, out)}``


---
## `lora_delta`

```python
def lora_delta(adapter: 'dict[str, Tensor]', alpha: 'float' = 1.0) -> 'Tensor':
```
Compute the scaled LoRA weight update: ``(alpha / rank) * A @ B``.

**Parameters**

- **`adapter`** – Dict with keys ``'A'`` ``(in, rank)`` and ``'B'`` ``(rank, out)``.
- **`alpha`** – Scaling factor. Default: ``1.0``.

**Returns**

Delta tensor of shape ``(in, out)``.


---
## `lora_linear`

```python
def lora_linear(x: 'Tensor', frozen_weight: 'Tensor', adapter: 'dict[str, Tensor]', alpha: 'float' = 1.0) -> 'Tensor':
```
Linear projection with frozen path + LoRA adapter path.


---
## `merge_lora_weight`

```python
def merge_lora_weight(frozen_weight: 'Tensor', adapter: 'dict[str, Tensor]', alpha: 'float' = 1.0) -> 'Tensor':
```
Merge the LoRA adapter into the frozen weight: ``W_merged = W + delta``.

**Parameters**

- **`frozen_weight`** – Original frozen weight tensor.
- **`adapter`** – LoRA adapter dict (see :func:`init_lora_adapter`).
- **`alpha`** – Scaling factor for the adapter. Default: ``1.0``.

**Returns**

Merged weight tensor with the same shape as *frozen_weight*.


---
## `unmerge_lora_weight`

```python
def unmerge_lora_weight(merged_weight: 'Tensor', adapter: 'dict[str, Tensor]', alpha: 'float' = 1.0) -> 'Tensor':
```
Recover the original frozen weight by subtracting the LoRA delta.

**Parameters**

- **`merged_weight`** – Previously merged weight tensor.
- **`adapter`** – LoRA adapter dict used during merging.
- **`alpha`** – Scaling factor used during merging. Default: ``1.0``.

**Returns**

Recovered frozen weight tensor.


---
## `tree_lora_delta`

```python
def tree_lora_delta(adapters: 'Any', alpha: 'float' = 1.0, *, is_leaf: 'Any' = None) -> 'Any':
```
Map a pytree of LoRA adapter dicts to their low-rank deltas.


---
