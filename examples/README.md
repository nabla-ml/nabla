# Nabla Examples

This folder contains runnable example `.py` files that mirror the tutorial notebooks in `docs/examples/`.

## Learning path

1. `01_tensors_and_ops.py` — Tensor creation, indexing, and basic operations
2. `02_autodiff.py` — Automatic differentiation: grad, value_and_grad, Jacobians, Hessians
3. `03a_mlp_training_pytorch.py` — MLP training (PyTorch-style `nn.Module`)
4. `03b_mlp_training_jax.py` — MLP training (JAX-style functional)
5. `04_transforms_and_compile.py` — `vmap`, `jacrev`, `jacfwd`, `@nb.compile`
6. `05a_transformer_pytorch.py` — Transformer classifier (PyTorch-style)
7. `05b_transformer_jax.py` — Transformer classifier (JAX-style functional)
8. `06_mlp_pipeline_parallel.py` — Pipeline-parallel training with GPipe
9. `07_mlp_pp_dp_training.py` — 2D parallelism (pipeline + data parallel)
10. `08_mlp_pipeline_inference.py` — Pipeline-parallel inference
11. `09_jax_comparison_compiled.py` — Benchmarking compiled Nabla vs JAX
12. `10_lora_finetuning_mvp.py` — LoRA & QLoRA fine-tuning

## Notes

- The notebook files in `docs/examples/` are the primary tutorial format and may contain additional content (graph tracing, custom Mojo kernels) beyond what is covered here.
- Some `.py` files still use old naming (03a/b, 04, 05a/b, …) — they correspond to notebooks 04a/b, 05, 06a/b, … respectively.
