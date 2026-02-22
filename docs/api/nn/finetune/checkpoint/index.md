# Checkpointing

## `save_finetune_checkpoint`

```python
def save_finetune_checkpoint(path: 'PathLike', *, lora_params: 'Any', optimizer_state: 'dict[str, Any] | None' = None, metadata: 'dict[str, Any] | None' = None) -> 'None':
```
Save LoRA params and optional optimizer state to Nabla-native checkpoint files.


---
## `load_finetune_checkpoint`

```python
def load_finetune_checkpoint(path: 'PathLike', *, lora_template: 'Any', optimizer_template: 'dict[str, Any] | None' = None) -> 'tuple[Any, dict[str, Any] | None, dict[str, Any]]':
```
Load LoRA params and optional optimizer state from checkpoint.


---
