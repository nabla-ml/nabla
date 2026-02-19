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

<style>
/* Homepage-only layout override */
.bd-sidebar-primary,
#pst-primary-sidebar-checkbox,
label[for="pst-primary-sidebar-checkbox"],
.primary-toggle {
	display: none !important;
}

.bd-container__inner.bd-page-width {
	grid-template-columns: minmax(0, 1fr) !important;
	max-width: 100% !important;
}

.bd-main {
	grid-column: 1 / -1 !important;
}

.bd-main .bd-content {
	padding-left: 0 !important;
}

.bd-main .bd-content .bd-article-container {
	max-width: 1200px !important;
	margin: 0 auto !important;
}
</style>

::::{div} landing-hero
# Nabla

**High-Performance Distributed ML**

Nabla is a JAX-inspired autodiff library with factor-based SPMD sharding, built on [Mojo & MAX](https://www.modular.com/max).

```bash
pip install --pre --extra-index-url https://whl.modular.com/nightly/simple/ modular nabla-ml
```
::::

::::{grid} 1 1 3 3
:gutter: 3
:class-container: landing-grid

:::{grid-item-card} 🚀 Get Started
:link: get_started
:link-type: doc
:text-align: center

Learn the basics of Nabla, from tensors and autodiff to SPMD sharding and Mojo integration.
:::

:::{grid-item-card} 📚 API Reference
:link: api/index
:link-type: doc
:text-align: center

Detailed documentation for Nabla's core modules, operations, and transforms.
:::

:::{grid-item-card} 💡 Examples
:link: examples/index
:link-type: doc
:text-align: center

Explore guided notebooks demonstrating training, pipeline parallelism, and fine-tuning.
:::
::::
