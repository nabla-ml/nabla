# Testing (nabla.testing)

Utilities for testing and validating tensor computations.

## `assert_allclose`

```python
def assert_allclose(actual: 'Any', expected: 'Any', *, rtol: 'float' = 1e-05, atol: 'float' = 1e-08, equal_nan: 'bool' = False, realize: 'bool' = True) -> 'None':
```
Assert numerical closeness across Nabla/JAX/PyTorch/NumPy objects.

By default realizes Nabla tensors first; pass `realize=False` if callers
already did an explicit `batch_realize(...)` for efficiency.


---
## `batch_realize`

```python
def batch_realize(*objs: 'Any') -> 'None':
```
Batch-realize all Nabla tensors found in arbitrary pytree-like objects.


---
