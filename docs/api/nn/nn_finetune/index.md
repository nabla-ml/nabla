# Fine-Tuning (nabla.nn.finetune)

Parameter-efficient fine-tuning with LoRA adapters and NF4 quantization (QLoRA).

## LoRA

`init_lora_adapter`, `apply_lora`, `merge_lora` — low-rank adapter training and merging.

```{toctree}
:maxdepth: 1
:hidden:

nn_finetune_lora/index
```

## QLoRA

`quantize_nf4`, `dequantize_nf4`, `qlora_linear` — NF4 quantization for memory-efficient fine-tuning.

```{toctree}
:maxdepth: 1
:hidden:

nn_finetune_qlora/index
```

## Checkpoint

Save and load adapter weights, full model checkpoints, and quantized states.

```{toctree}
:maxdepth: 1
:hidden:

nn_finetune_checkpoint/index
```
