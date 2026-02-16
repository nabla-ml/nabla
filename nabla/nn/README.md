# Nabla NN Strategy

Nabla `nn` supports two complementary usage styles:

- **Imperative module style** (`nb.nn.Module`, `nb.nn.Linear`, `nb.nn.Sequential`) for PyTorch-like workflows.
- **Functional style** (`nb.nn.functional.*`, `nb.grad`, `nb.value_and_grad`) for pure-transform workflows (JAX/Equinox-like).

## MAX Adapter Surface

`nb.nn.adapt_max_module_class` and `nb.nn.adapt_max_nn_core` provide compatibility adapters for selected `max.nn` classes.

Current validated core set:

- `Linear`
- `Embedding`
- `ModuleList`
- `Sequential`

The adapter is intended for **progressive interoperability**, not as a replacement for high-quality native Nabla modules.

## What We Keep

- Native Nabla modules as the canonical long-term API.
- Functional-first training/update path as a first-class interface.
- MAX adapters where migration value is high and behavior is covered by tests.

## What We Avoid

- Shipping placeholder/fake compatibility layers in production code.
- Expanding adapter coverage without real-module tests.
- Duplicating native module implementations just because an adapted variant exists.

## Testing Placement

Adapter tests live under `tests/unit/nn/`:

- `test_max_adapter_real.py` for real class adaptation behavior.
- `test_max_adapter_training.py` for end-to-end training with Nabla optimizer.

Add new adapter coverage in this directory and gate by `pytest.importorskip("max.nn")` when needed.
