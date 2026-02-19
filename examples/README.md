# Nabla Examples

This folder is the single source of truth for runnable tutorial-style `.py` examples.

## Learning path

1. `01_tensors_and_ops.py`
2. `02_autodiff.py`
3. `03a_mlp_training_pytorch.py`
4. `03b_mlp_training_jax.py`
5. `04_transforms_and_compile.py`
6. `05a_transformer_pytorch.py`
7. `05b_transformer_jax.py`
8. `06_mlp_pipeline_parallel.py`
9. `07_mlp_pp_dp_training.py`
10. `08_mlp_pipeline_inference.py`
11. `09_jax_comparison_compiled.py`
12. `10_lora_finetuning_mvp.py`
13. `11_qlora_finetuning_mvp.py`

## Notes

- The notebook tutorials in `docs/tutorials/` are generated from the first 7 files above.
- Run the converter with:

```bash
python tutorials/_convert_to_notebooks.py
```
