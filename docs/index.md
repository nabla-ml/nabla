---
hide-toc: true
html_theme.sidebar_secondary.remove: true
---

```{toctree}
:maxdepth: 2
:hidden:

get_started
examples/index
api/index
```

::::{div} landing-hero
<h1 class="landing-title">PyTorch Ease. JAX Scale. Mojo Speed.</h1>
<p class="landing-subtitle">Nabla is a modular scientific computing and machine learning framework combining imperative and functional APIs. Seamlessly drop custom Mojo kernels into the autodiff engine and automatically shard distributed workloads.</p>

::::{div} landing-install
```bash
pip install --pre --extra-index-url https://whl.modular.com/nightly/simple/ modular nabla-ml
```
::::
::::

::::{grid} 1 1 3 3
:gutter: 4
:class-container: landing-grid-modular

:::{grid-item-card}
:link: get_started
:link-type: doc
:class-card: home-nav-card

<div class="card-image-container">
    <img src="_static/images/cards/get-started.png" alt="Get started in Nabla" class="card-media" />
</div>
<div class="card-title-bottom">Getting Started</div>
<div class="card-subtitle">Setup, tensors, autodiff, and your first training loop.</div>
:::

:::{grid-item-card}
:link: api/index
:link-type: doc
:class-card: home-nav-card

<div class="card-image-container">
    <img src="_static/images/cards/api.png" alt="Nabla API reference" class="card-media" />
</div>
<div class="card-title-bottom">API Reference</div>
<div class="card-subtitle">Core tensors, transforms, ops, and neural-network primitives.</div>
:::

:::{grid-item-card}
:link: examples/index
:link-type: doc
:class-card: home-nav-card

<div class="card-image-container">
    <img src="_static/images/cards/examples.png" alt="Nabla examples and tutorials" class="card-media" />
</div>
<div class="card-title-bottom">Examples</div>
<div class="card-subtitle">Guided notebooks across MLPs, transformers, and fine-tuning.</div>
:::
::::

## Why Nabla

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Eager Metadata, Compiled Speed
:class-card: info-card

Get the debuggability of PyTorch with the performance of compiled graphs. Nabla computes shapes eagerly but defers graph building, using zero-overhead caching to skip recompilation on hot paths.
:::

:::{grid-item-card} Automatic Sharding (SPMD)
:class-card: info-card

Write single-device code and let Nabla handle the rest. Nabla automatically propagates sharding specs and handles communication for distributed training and inference workloads.
:::

:::{grid-item-card} Custom Mojo Kernels in Autodiff
:class-card: info-card

Need more performance? Drop down to Mojo for custom kernels and seamlessly integrate them into Nabla's autodiff engine via `nabla.call_custom_kernel(...)`.
:::

:::{grid-item-card} Modular by design
:class-card: info-card

Use `nb.nn.Module` and `nb.nn.functional` side-by-side. Nabla supports imperative and functional workflows in one framework, so you can use the style that fits your workflow.
:::
::::

## Try It Quickly

::::{tab-set}
:::{tab-item} Imperative style (PyTorch-like)
```python
import nabla as nb

model = nb.nn.Sequential(
    nb.nn.Linear(128, 256),
    nb.nn.ReLU(),
    nb.nn.Linear(256, 10),
)
```
:::

:::{tab-item} Functional style (JAX-like)
```python
import nabla as nb


def loss_fn(x, w):
    return nb.mean(nb.relu(x @ w))


grad_w = nb.grad(loss_fn, argnums=1)(x, w)
```
:::

:::{tab-item} QLoRA Fine-Tuning
```python
import nabla as nb

# Quantize frozen weights to NF4
qweight = nb.nn.finetune.quantize_nf4(frozen_weight, block_size=64)

# Initialize LoRA adapter
lora_params = nb.nn.finetune.init_lora_adapter(frozen_weight, rank=8)

# Forward pass with QLoRA
def loss_fn(adapter, batch_x, batch_y):
    pred = nb.nn.finetune.qlora_linear(
        batch_x, qweight, adapter, alpha=16.0
    )
    return nb.mean((pred - batch_y) ** 2)

# Compute gradients
loss, grads = nb.value_and_grad(loss_fn)(lora_params, x, y)
```
:::
::::

::::{div} featured-example
### Featured Example
[LoRA fine-tuning MVP →](examples/10_lora_finetuning_mvp)

See a compact end-to-end fine-tuning flow in the examples collection.
::::

::::{div} blog-slot
### From the team
Blog and release notes section coming soon.
::::

## Project Status

Nabla is currently in **Alpha**. It is an experimental framework designed to explore new ideas in ML infrastructure on top of Modular MAX. APIs are subject to change, and we welcome early adopters to join us in building the next generation of ML tools.

