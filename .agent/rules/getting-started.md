---
trigger: always_on
---

# Nabla: Development Guide

> [!IMPORTANT]
> **Environment Activation**: Ensure your virtual environment is active before running code:
> ```bash
> source venv/bin/activate
> ```

Definitive technical reference for Nabla. It reflects the `nabla` module hierarchy, indexing the codebase, data structures, and the lazy-SPMD execution model.

---

## 1. Package Entry & Public API ([`nabla/`](nabla/))

Root of the package containing public API entry points.

### Component Map
*   [`__init__.py`](nabla/__init__.py): Entry point. Re-exports the user-facing API (`Tensor`, `trace`, `compile`, `vmap`, `shard_map`) for a clean namespace.
*   [`README.md`](nabla/README.md): High-level framework overview.

---

## 2. Core Engine ([`nabla/core/`](nabla/core/))

The `core` module houses Nabla's theoretical foundations and execution engine, covering utilities, data structures, graph management, and sharding.

### 2.1 Common Utilities ([`nabla/core/common/`](nabla/core/common/))
Shared logic across the framework.

*   [`context.py`](nabla/core/common/context.py): **Global State**. Uses `ContextVar` for thread-safe session management.
    *   **Context Defaults**: Resolves context-based defaults (e.g., `dtype`, `device`) for tensor creation.
*   [`pytree.py`](nabla/core/common/pytree.py): **PyTree Stack**. JAX-like recursive structures.
    *   Handles nested dictionaries/lists of tensors via `flatten` and `unflatten`.

### 2.2 The Tensor System ([`nabla/core/tensor/`](nabla/core/tensor/))
Primary data structures for users and the engine.

*   [`api.py`](nabla/core/tensor/api.py): **`Tensor` API**. User-facing proxy class.
*   [`impl.py`](nabla/core/tensor/impl.py): **`TensorImpl` Inner Class**. Critical node in the computation graph.
    *   **Internal State**:
        *   `_values`: Lazy `graph.Value` objects.
        *   `_storages`: Realized physical data (`driver.Tensor`).
        *   `sharding`: `ShardingSpec` layout.
        *   `batch_dims`: Physical dimensions for `vmap`.
        *   `output_refs`: Metadata container pointing to parent `Operation`.
    *   **Properties**: `global_shape` (logical), `physical_shape` (local), `is_realized`.
    *   **Methods**: `realize()`, `to_numpy()`, `item()`, `to_dlpack()`.

### 2.3 Lazy Graph & Tracing ([`nabla/core/graph/`](nabla/core/graph/))
Manages lazy evaluation and functional tracing.

*   [`engine.py`](nabla/core/graph/engine.py): **`ComputeGraph` (GRAPH)**. Manages unrealized tensors.
*   [`tracing.py`](nabla/core/graph/tracing.py): **Tracing Infrastructure**.
    *   **`OutputRefs`**: Shared metadata for sibling outputs. Uses weak references to avoid cycles.
    *   **`trace()`**: Captures magical mathematical structure.
*   [`utils.py`](nabla/core/graph/utils.py): Internal graph-walking helpers.

#### The Execution Loop (`execute_operation`)
Located in [`nabla/ops/dispatch.py`](nabla/ops/dispatch.py), Nabla's "Eagerly-Lazy" heart:
1.  **Metadata**: Inspects input `traced` flags, `batch_dims`, and `sharding`.
2.  **Mesh**: Determines the target `DeviceMesh`.
3.  **Inference**: Calls `sharding_rule` to predict output layout.
4.  **Resharding**: Injects resharding operations if inputs don't match the inferred rule.
5.  **Shard Loop**: Iterates through physical shards, calling the op's `maxpr` method.
6.  **Post-Process**: Applies `AllReduce` for sharded contracting dims and propagates AD via `jvp_rule`.

### 2.4 Sharding & SPMD ([`nabla/core/sharding/`](nabla/core/sharding/))
Calculates sharding strategies for distributed execution. Inspired by **XLA Shardy (Sdy)**, using factor-based IR for propagation and automatic AllReduce.

#### The Sharding IR: Factors
Translates tensor dimensions into **Factors** ([`propagation.py`](nabla/core/sharding/propagation.py)).
- **Factors** are logical sharding units (e.g., `m`, `k`, `n` in `A[m,k] @ B[k,n]`).
- Propagation at factor level decouples execution from specific tensor ranks or shapes.

#### Hierarchical Propagation Loop
Resolves constraints through a multi-tiered loop:
1.  **User Priorities**: Propagates specs (`p0` to `p10`). Stronger priorities act as anchors.
2.  **Op Priorities**: Groups ops by structure:
    - **PASSTHROUGH**: (Elementwise/Reshape) High priority.
    - **CONTRACTION**: (Matmuls/Einsums) Moderate priority.
    - **REDUCTION**: Propagated after contractions.
3.  **Strategy**:
    - **AGGRESSIVE**: Maximizes parallelism.
    - **BASIC**: Uses "Common Prefix" matching (conservative).

#### Sharding Rules & Einsum
Ops define logic via `sharding_rule()` using einsum syntax (e.g., `m k, k n -> m n`).
- **Input/Output Mappings**: Factors assigned to dimensions.
- **Contracting Factors**: Factors in inputs but absent from output (e.g., `k`).
- **Auto-AllReduce**: If mesh axes map to a **Contracting Factor**, the SPMD pipeline injects `AllReduce` to aggregate partial sums.

#### SPMD Pipeline ([`spmd.py`](nabla/core/sharding/spmd.py))
Transforms a logical graph into a distributed one:
1.  **Inference**: Projects shardings through the graph.
2.  **Smart Resharding**: Injects `AllGather`, `AllReduce`, or `Reshard` only if the new spec isn't a strict extension.
3.  **Hydration**: "Hydrates" lazy tensors into physical data on-device when concrete values are required.
4.  **Shard Creation**: Wraps physical `driver.Tensor` shards into a logical `TensorImpl`.

---

## 3. Operations Library ([`nabla/ops/`](nabla/ops/))

Stateless singletons defining graph building, sharding, and derivatives.

### 3.1 Base & Dispatch
*   [`base.py`](nabla/ops/base.py): **`Operation` Interface**.
    *   **`maxpr(*args, **kwargs)`**: Emits raw MAX graph instructions.
    *   **`sharding_rule(...)`**: Defines factor-based SPMD propagation.
    *   **`vjp_rule` / `jvp_rule`**: Rules for Reverse and Forward AD.
    *   **`compute_cost` / `memory_cost`**: Metadata for the sharding solver.
    *   **`infer_output_rank`**: Required for lazy tracing where physical shapes are unknown.

#### The `__call__` Structure
Manages transition from logical requests to physical nodes:
1.  **Normalization**: Ensures inputs are `Tensor` objects.
2.  **Translation**:
    *   **`LogicalAxisOperation`**: Offsets user axes by `batch_dims`.
    *   **`LogicalShapeOperation`**: Prepends physical batch shape to global shapes.
    *   **`BinaryOperation`**: Handles broadcasting for Logical and Physical (vmap) dims.
3.  **Dispatch**: Calls `execute_operation` to trigger SPMD.

### 3.2 Math Operations ("Verbs")
*   [`unary.py`](nabla/ops/unary.py), [`binary.py`](nabla/ops/binary.py), [`reduction.py`](nabla/ops/reduction.py), [`creation.py`](nabla/ops/creation.py), [`comparison.py`](nabla/ops/comparison.py).

### 3.3 Graph Control & Multi-Output
*   [`multi_output.py`](nabla/ops/multi_output.py), [`control_flow.py`](nabla/ops/control_flow.py).

### 3.4 View & Metadata Ops ([`nabla/ops/view/`](nabla/ops/view/))
Metadata-only manipulation: [`axes.py`](nabla/ops/view/axes.py), [`shape.py`](nabla/ops/view/shape.py), [`indexing.py`](nabla/ops/view/indexing.py).

### 3.5 Communication Collectives ([`nabla/ops/communication/`](nabla/ops/communication/))
Kernels for data movement: [`all_reduce.py`](nabla/ops/communication/all_reduce.py), [`all_gather.py`](nabla/ops/communication/all_gather.py).

---

## 4. Function Transforms ([`nabla/transforms/`](nabla/transforms/))

### 4.1 `vmap` (Vectorization) ([`vmap.py`](nabla/transforms/vmap.py))
- Uses `batch_dims` on `TensorImpl`.
- `_batch_tensor` lifts and moves specified axis to physical position 0.
- Ops handle the extra dimension via `BinaryOperation` or `LogicalAxisOperation`.
- `_unbatch_tensor` restores logical position on exit.

### 4.2 `shard_map` (Parallelization) ([`shard_map.py`](nabla/transforms/shard_map.py))
- Automatic SPMD via ILP:
1.  **Tracing**: Captures logical DAG.
2.  **Extraction**: Outputs JSON graph description.
3.  **Solving**: Runs `SimpleSolver` (ILP) for minimal communication sharding.
4.  **Replay**: Re-executes logic with `shard()` constraints. Eager-resharding handles the rest.

### 4.3 `compile` (JIT) ([`compile.py`](nabla/transforms/compile.py))
- Converts Python to optimized MAX Mojo models:
1.  **Proxy Tracing**: Uses `Tensor` proxies wrapping `graph.SymbolicValue`.
2.  **Tracing**: Builds a MAX Graph.
3.  **Symbolic Dims**: Uses `dynamic_dims` for variable batch sizes.
4.  **Caching**: Models cached by `_CacheKey` (ranks, dtypes, pytree structure).

---

## 5. Testing & Validation

Nabla uses "Baseline Verification" ensuring consistency with NumPy.

### Pattern
1.  **Preparation**: Create NumPy reference data.
2.  **Lifting**: Convert to Nabla `Tensor` (`from_dlpack`).
3.  **Execution**: Run Nabla logic.
4.  **Observation**: (Optional) Use `trace().print()` to inspect the graph.
5.  **Validation**: Call `.to_numpy()` and compare via `np.testing.assert_allclose`.

### Example
```python
import numpy as np
import nabla

# 1. Inputs
np_a = np.random.randn(8, 4).astype(np.float32)
np_b = np.random.randn(8, 4).astype(np.float32)

# 2. Lift
a = nabla.Tensor.from_dlpack(np_a)
b = nabla.Tensor.from_dlpack(np_b)

# 3. Execute
def func(a, b): return a + b
result = func(a, b)

# 4. Observe
# print(nabla.trace(func, a, b))

# 5. Validate
actual = result.to_numpy()
expected = np_a + np_b
np.testing.assert_allclose(actual, expected, rtol=1e-5)
```

### Infrastructure Helpers ([`tests/conftest.py`](tests/conftest.py))
- `tensor_from_numpy(arr)`: `from_dlpack` wrapper.
- `to_numpy(tensor)`: Grounds lazy/sharded tensors.
- `assert_allclose(result, expected)`: Automated comparison.
