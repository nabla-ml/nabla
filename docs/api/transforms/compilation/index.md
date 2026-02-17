# Compilation

## `compile`

```python
def compile(fn: 'Callable[..., T] | None' = None, *, fullgraph: 'bool' = False, max_cache_size: 'int' = 64, dynamic_dims: 'dict[int, dict[int, str]] | None' = None) -> 'CompiledFunction[T] | Callable[..., CompiledFunction[Any]]':
```
Compile a function for cached graph execution.

**Parameters**

- **`fn`** – Function to compile. If None, returns a decorator.
- **`fullgraph`** – If True, error on side effects. If False, fall back to eager.
- **`max_cache_size`** – Maximum cached compilations (LRU eviction).
- **`dynamic_dims`** – Mark dimensions as symbolic. Format: {arg_idx: {dim_idx: "name"}}
E.g., {0: {0: "batch"}} makes arg 0, dim 0 dynamic.

**Returns**

 – CompiledFunction wrapping the original function.

**Examples**

*Warning: Could not parse examples correctly.*

---
## `CompiledFunction`

```python
class CompiledFunction(fn: 'Callable[..., T]', *, fullgraph: 'bool' = False, max_cache_size: 'int' = 64, dynamic_dims: 'dict[int, dict[int, str]] | None' = None):
```
A compiled function with caching.


---
