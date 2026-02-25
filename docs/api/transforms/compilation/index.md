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

CompiledFunction wrapping the original function.

**Examples**

*Warning: Could not parse examples correctly.*

---
## `CompiledFunction`

```python
class CompiledFunction(fn: 'Callable[..., T]', *, fullgraph: 'bool' = False, max_cache_size: 'int' = 64, dynamic_dims: 'dict[int, dict[int, str]] | None' = None):
```
A JIT-compiled function with signature-based LRU caching.

On each call, the argument signatures (shapes, dtypes, pytree structure)
are hashed and looked up in the cache. On a cache hit the pre-compiled
MAX graph model is executed directly; on a miss the function is traced
and compiled before execution.

**Parameters**

- **`stats`** – A :class:`CompilationStats` instance tracking hits, misses,
fallbacks, and compile time.


---
## `CompilationStats`

```python
class CompilationStats(hits: 'int' = 0, misses: 'int' = 0, fallbacks: 'int' = 0, total_compile_time_ms: 'float' = 0.0, total_cached_exec_time_ms: 'float' = 0.0, cache_size: 'int' = 0) -> None:
```
Runtime statistics for a :class:`CompiledFunction`.

**Parameters**

- **`hits`** – Number of cache hit executions (fast path).
- **`misses`** – Number of cache misses that triggered recompilation.
- **`fallbacks`** – Number of calls that fell back to eager execution.
- **`total_compile_time_ms`** – Cumulative compilation time in milliseconds.
- **`total_cached_exec_time_ms`** – Cumulative execution time for cache hits.
- **`cache_size`** – Current number of entries in the LRU cache.


---
