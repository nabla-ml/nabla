# Function Transforms

[← Back to Root](../CLAUDE.md)

Wrapping functions to change their behavior.

## The Transforms

| Transform | Purpose | Key File | Mechanism |
| :--- | :--- | :--- | :--- |
| **[`vmap`](vmap.py)** | **Vectorization**. Auto-batches operations. | [`vmap.py`](vmap.py) | **Prefix Semantics**. Batch dims are always physically leading. |
| **[`shard_map`](shard_map.py)** | **Distribution**. Logical code -> Physical execution. | [`shard_map.py`](shard_map.py) | **Dual Execution**. Trace logical, replay physical with SPMD. |
| **[`compile`](compile.py)** | **Optimization**. JIT caching & side-effect checks. | [`compile.py`](compile.py) | **Shape Caching**. Reuses graphs for dynamic batch sizes. |

## 1. VMap (Vectorizing Map)
`vmap` transforms a function operating on single examples to one operating on batches.

```python
vmapped_fn = vmap(fn, in_axes=(0, None), out_axes=0)
```

### Prefix Semantics
Unlike JAX, we enforce **Physical Prefix Semantics**:
-   **Batch Dims**: Always the *leading* physical dimensions.
-   **Logical Shape**: What the user function sees (batch dims hidden).
-   **Propagation**: Binary ops merge batch dims: `max(x.dims, y.dims)`.

## 2. Shard Map (Manual Distribution)
`shard_map` allows writing logical single-device code but executing it across a mesh.

```python
# User writes logical code
def log_fn(x, w): return x @ w

# We map it to physical execution
phys_fn = shard_map(
    log_fn, mesh,
    in_specs={0: P('dp'), 1: P('tp')},
    out_specs=P('dp', 'tp')
)
```

### The Trace-and-Replay Model
1.  **Trace**: We run `log_fn` with logical tensors to capture the graph.
2.  **Replay**: We re-execute the graph using physical tensors (Duals).
3.  **Result**: A sharded execution plan without complex graph patching.

> **⚠️ CRITICAL**: Do NOT call `shard()` inside a `shard_map` function. Specify sharding via `in_specs`.

## 3. Compile (JIT)
`compile` caches the MAX graph to avoid Python overhead.

```python
@compile(dynamic_dims={0: {0: "batch"}})
def f(x): ...
```

-   **Lazy by Default**: Nabla is already lazy-eager. `compile` is optional.
-   **Dynamic Shapes**: Use `dynamic_dims` to compile once for variable batch sizes.
-   **Strict Mode**: `fullgraph=True` checks for side effects (like `print` or `.numpy()`) via epoch tracking.
