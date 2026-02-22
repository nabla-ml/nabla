# LoRA

## `init_lora_adapter`

```python
def init_lora_adapter(weight: 'Tensor', rank: 'int', init_std: 'float' = 0.01, dtype: 'DType | None' = None) -> 'dict[str, Tensor]':
```
Initialize LoRA adapter matrices for a 2D linear weight.


---
## `lora_delta`

```python
def lora_delta(adapter: 'dict[str, Tensor]', alpha: 'float' = 1.0) -> 'Tensor':
```
Compute scaled LoRA low-rank delta: (alpha / rank) * (A @ B).


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
Return merged weight: W + (alpha/r) * A @ B.


---
## `unmerge_lora_weight`

```python
def unmerge_lora_weight(merged_weight: 'Tensor', adapter: 'dict[str, Tensor]', alpha: 'float' = 1.0) -> 'Tensor':
```
Recover frozen weight from merged weight and adapter.


---
## `tree_lora_delta`

```python
def tree_lora_delta(adapters: 'Any', alpha: 'float' = 1.0, *, is_leaf: 'Any' = None) -> 'Any':
```
Map a pytree of LoRA adapter dicts to their low-rank deltas.


---
