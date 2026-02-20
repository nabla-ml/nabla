---
hide-toc: true
---

```{toctree}
:maxdepth: 2
:hidden:

get_started
examples/index
api/index
```

::::{div} landing-hero
<h1 class="landing-title">Modular ML Framework for Autodiff, SPMD, and Custom Kernels</h1>
<p class="landing-subtitle">Nabla gives you both imperative (PyTorch-like) and functional (JAX-like) APIs, so you can use the style that fits your workflow without being locked into one paradigm.</p>

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

<img src="_static/images/cards/get-started.png" alt="Get started in Nabla" class="card-media" />
<div class="card-title-bottom">Getting Started</div>
<div class="card-subtitle">Setup, first tensors, autodiff, and your first training loop.</div>
:::

:::{grid-item-card}
:link: api/index
:link-type: doc
:class-card: home-nav-card

<img src="_static/images/cards/api.png" alt="Nabla API reference" class="card-media" />
<div class="card-title-bottom">API Reference</div>
<div class="card-subtitle">Core tensors, transforms, ops, and neural-network primitives.</div>
:::

:::{grid-item-card}
:link: examples/index
:link-type: doc
:class-card: home-nav-card

<img src="_static/images/cards/examples.png" alt="Nabla examples and tutorials" class="card-media" />
<div class="card-title-bottom">Examples</div>
<div class="card-subtitle">Guided notebooks across MLPs, transformers, and fine-tuning.</div>
:::
::::

# Why Nabla

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Modular by design
:class-card: info-card

Use `nb.nn.Module` and `nb.nn.functional` side-by-side. Nabla supports imperative and functional workflows in one framework.
:::

:::{grid-item-card} Cross-vendor GPU support
:class-card: info-card

Documented support via Modular MAX for Linux (AMD/NVIDIA) and macOS Apple Silicon (with Metal toolchain).
:::

:::{grid-item-card} Custom Mojo kernels
:class-card: info-card

Drop down to Mojo for custom kernels and call them from Python via `nabla.call_custom_kernel(...)`.
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
