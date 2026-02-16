# Nabla NN Strategy

Nabla `nn` supports two complementary usage styles:

- **Imperative module style** (`nb.nn.Module`, `nb.nn.Linear`, `nb.nn.Sequential`) for PyTorch-like workflows.
- **Functional style** (`nb.nn.functional.*`, `nb.grad`, `nb.value_and_grad`) for pure-transform workflows (JAX/Equinox-like).

## What We Keep

- Native Nabla modules as the canonical long-term API.
- Functional-first training/update path as a first-class interface.

## What We Avoid

- Shipping compatibility layers that increase complexity without long-term ownership.
- Duplicating module implementations under parallel APIs.

## Testing Placement

All native `nn` coverage lives under `tests/unit/nn/`.

For new module work:

- Add unit tests for forward/backward semantics.
- Add transform coverage where relevant (`value_and_grad`, `vmap`, `compile`).
- Prefer deterministic toy-training tests for optimizer integration.
