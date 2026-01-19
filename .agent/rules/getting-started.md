---
trigger: always_on
---

# Nabla Architecture

> **CRITICAL**: Always activate the virtual environment first:
> `source venv/bin/activate`
>
> **MANDATORY**: You **MUST** read all `README.md` files in the `nabla` module recursively to understand the codebase. You are **NOT ALLOWED** to bypass this step.

## Philosophy
Nabla is a **Lazy-eager** tensor execution framework for MAX. It combines:
1.  **Imperative API**: A PyTorch-like interface for user-friendly coding.
2.  **Lazy Graph**: Operations are traced into a `ComputeGraph`, allowing global optimization.
3.  **Unified SPMD**: A "Single Program, Multiple Data" compiler that automatically distributes code across devices using factor-based sharding propagation.

## Architecture Layers
The project is structured hierarchically to prevent circular dependencies.

| Layer | Module | Description | Guide |
| :--- | :--- | :--- | :--- |
| **User** | **`nabla/`** | The public API surface. | [Read Here](nabla/README.md) |
| **Logic** | **`nabla/ops/`** | Operation definitions and dispatch. | [Read Here](nabla/ops/README.md) |
| **State** | **`nabla/core/`** | The engine: Tensors, Graphs, Sharding. | [Read Here](nabla/core/README.md) |
| **Glue** | **`nabla/transforms/`** | Functional transforms (vmap/shard_map/compile). | [Read Here](nabla/transforms/README.md) |

## Testing Strategy
We test both the "Logical" (single-device correctness) and "Physical" (distributed splitting) behaviors.
[**Read the Testing Guide**](tests/README.md)

## Key Workflows
*   **Running Tests**: `python -m pytest tests`

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Scope**: This is the Root Documentation for AI Agents. Use the links in the table above to guide your exploration.
> 2.  **Update Requirement**: You **MUST** update this file whenever you modify the high-level project structure, add top-level modules, or change the "Philosophy".
> 3.  **Standards**: All `README.md` files must strictly follow the [Template: Philosophy -> Architecture -> Map -> Maintenance].
> 4.  **Creation**: If you create a new module, you **MUST** create a corresponding `README.md` immediately.

## Repository Map

```
nabla  # Root package
├── README.md
├── __init__.py
├── core  # The core engine modules (Tensors, Graphs, Sharding)
│   ├── README.md
│   ├── __init__.py
│   ├── common
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── context.py  # Global configuration state (Default Device/DType) and a cached, thread-safe session singleton for compiling/running MAX kernels.
│   │   └── pytree.py  # High-throughput tree flattening/unflattening using JAX-style registry. Optimized for single-pass mapping and handles custom nodes/NamedTuples.
│   ├── graph
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── engine.py  # `ComputeGraph` singleton. The execution loop: lazily accumulates nodes in `unrealized`, compiles graphs in `evaluate` when data is requested, and manages the MLIR module lifecycle.
│   │   ├── tracing.py  # Defines `Trace` and `OutputRefs`. Uses `weakref` to track provenance. IMPORTANT: `Trace` captures the graph boundary, while `OutputRefs` are the graph nodes held by Tensors. Tracing does only happen operations, where at least one of the inputs is marked as traced. This does not interfere with the lazy execiton system in any way. It is useful for specific transformations like debugging, trafos like shard_map, or future backpropation (gradient compuation) implementation. OutputRefs are only generated and stored per Operation for the resulting TensorImpl(s), if any of the arguments of the operaiton has been traced. That lets us later traferse our captured Graph, OutputRef per OutputRef, irrespective weather evaluations have "kicked in" or not.
│   │   └── utils.py  # Graph traversal algorithms (Topological Sort, DFS) used for VJP (backprop) and visualization/printing.
│   ├── sharding
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── optimizer
│   │   │   ├── README.md
│   │   │   └── simple_solver.py  # ILP/Heuristic solver for automatic sharding strategy.
│   │   ├── propagation.py  # Factor-based sharding propagation. Maps axis names -> Factors (i,j) -> Output. Resolves conflicts using priority (User > Conctraction > ...).
│   │   ├── spec.py  # Data structures: `DeviceMesh` (grid), `ShardingSpec` (tensor layout), `DimSpec` (axis mapping).
│   │   └── spmd.py  # The SPMD Pipeline: 1. Infer Specs (Factors) -> 2. Propagate -> 3. Reshard Inputs (Diff) -> 4. Generate Sharded Output.
│   └── tensor
│       ├── README.md
│       ├── __init__.py
│       ├── api.py  # User-facing `Tensor` facade. Immutable-ish wrapper.
│       └── impl.py  # `TensorImpl` (The STATE). Two states: 1. Lazy (`_values`: graph nodes), 2. Realized (`_storages`: buffers). Holds `sharding` metadata and `output_refs` for backward traversal.
├── ops
│   ├── README.md
│   ├── __init__.py
│   ├── base.py  # `Operation` ABC. Defines the interface for all ops (`maxpr`, `compute_cost`). Includes specialized subclasses like `BinaryOperation` (auto-broadcasting) and `ReduceOperation` (auto-resharding).
│   ├── binary.py
│   ├── communication
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── all_gather.py
│   │   ├── all_reduce.py
│   │   ├── all_to_all.py
│   │   ├── axis_index.py
│   │   ├── base.py
│   │   ├── p_permute.py
│   │   ├── reduce_scatter.py
│   │   ├── reshard.py
│   │   └── shard.py
│   ├── comparison.py
│   ├── control_flow.py
│   ├── creation.py
│   ├── custom_op.py
│   ├── dispatch.py  # The Unified Dispatcher (`execute_operation`). Handles the end-to-end flow: Collect Metadata -> Infer Sharding -> Reshard Inputs -> Execute `maxpr` per shard -> Reassemble Outputs -> Auto-insert AllReduce. However, most ops override the __call__ method completely, they follow the same setup though!
│   ├── multi_output.py
│   ├── reduction.py
│   ├── unary.py
│   └── view
│       ├── README.md
│       ├── __init__.py
│       ├── axes.py
│       ├── batch.py
│       ├── indexing.py
│       └── shape.py
└── transforms  # Functional transformations
    ├── README.md
    ├── __init__.py
    ├── compile.py  # JIT Compilation. Encapsulates `CompiledFunction`. Caches binaries based on input signatures (shape/dtype/static values). Uses `asyncio.run` to drive the MAX compiler.
    ├── shard_map.py  # SPMD `shard_map`. Wraps function with `trace` to capture logical graph. Extracts `json` for solver (if auto_sharding). Replays ops using `dual` tensors (Physical Tensors) to inject collectives (all_gather/shard).
    └── vmap.py  # `vmap`. Auto-vectorization via batch dimension pushing. Transforms logical axes to physical batch dims. Handles nested vmap by stacking batch dims at the front of the physical shape.
```