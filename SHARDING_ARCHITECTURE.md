# Nabla Sharding Architecture

This document outlines the architectural principles and execution flow of Nabla's distributed tensor system. It serves as the reference for how sharding *should* work, guiding future improvements and new operation implementations.

## Core Philosophy

Nabla implements a **Unified SPMD (Single Program, Multiple Data)** architecture. This means:

1.  **Uniformity**: There is no separate code path for "sharded" vs "unsharded" execution. Unsharded execution is simply a special case where the device mesh has size 1 (or is None).
2.  **Symbolic Propagation**: We define how sharding propagates using symbolic *factors* (like `batch`, `model_dim`) rather than concrete dimension indices. This decouples the logic from specific tensor shapes.
3.  **Lazy Evaluation**: The graph is built first, with sharding constraints propagated symbolically. Actual execution (and communication) happens only when the graph is realized.

---

## 2. The Unified Execution Pipeline

Every operation in Nabla (`Operation.__call__`) follows a strict 6-step pipeline. This ensures consistency across all distributed operations.

### Step 1: Metadata Collection
The system scans inputs to detect:
*   **Traced Tensors**: Are we building a graph or executing eagerly?
*   **Sharded Tensors**: Is there a `DeviceMesh` involved?
*   **Batch Dimensions**: For `vmap` support.

### Step 2: Sharding Inference (Propagation)
Before any data is touched, we infer sharding layout constraints.
*   **Input**: Logical shapes of inputs + Operation semantics.
*   **Process**:
    1.  **Template Instantiation**: We map concrete input shapes to abstract sharding factors (e.g., mapping a `(1024, 512)` input to `(batch, model_dim)`).
    2.  **Factor Propagation**: We solve for the optimal sharding layout using the `OpShardingRule`. (See "Factor-Based Propagation" below).
*   **Output**: A `ShardingSpec` for the output and required `ShardingSpecs` for all inputs.

### Step 3: Input Alignment (Resharding)
We compare the *actual* sharding of inputs with the *required* sharding from Step 2.
*   **Mismatch**: If an input is sharded on axis `x` but the operation requires it to be replicated (or sharded on `y`), we insert communication ops.
*   **Strategy**: "Gather-then-Shard". We gather incorrect dimensions to replicate them, then slice/shard them to match the target.

### Step 4: Local Execution (SPMD Loop)
We execute the operation locally on each shard.
*   **Iterate**: Loop `0` to `num_shards`.
*   **Slice**: Extract the local chunk of data for the current shard.
    *   *Sharded Inputs*: Use the pre-existing chunk.
    *   *Replicated Inputs*: Slice the global tensor to match the shard's view.
*   **Compute**: Run the core math operation (`maxpr`) on these local chunks.

### Step 5: Contracting Dimension Reduction
If the operation involved summing over a sharded dimension (e.g., a Matrix Multiply where the contracting dimension `k` was sharded), the partial results on each device are incomplete.
*   **Action**: Insert an `AllReduce` (sum) cross-device communication to synchronize the results.

### Step 6: Output Construction
The local results are re-assembled into a logical `Tensor`.
*   **Logical View**: The user sees a single `Tensor` object.
*   **Physical Reality**: The tensor holds a list of `TensorValues` (one per shard) or a set of distributed `DriverTensors`.

---

## 3. Factor-Based Propagation Theory

We use a factor-based system inspired by XLA Shardy to handle complex operations like `Matmul` generic commands without hardcoding dimension indices.

### The Problem
In a Matmul `C = A @ B`:
*   `A` has shape `(M, K)`
*   `B` has shape `(K, N)`
*   `C` has shape `(M, N)`

If we shard `A` on axis `data`, does that mean we shard `B` on axis `data`?
*   If we shard `A`'s dim 0 (`M`), `B` doesn't have an `M` dimension. `B` should be replicated (or sharded on `N`).
*   If we shard `A`'s dim 1 (`K`), `B`'s dim 0 is also `K`. They *must* verify compatibility.

### The Solution: Factors
We assign names to these dimensions: `m`, `k`, `n`.
The rule is defined as: `(m, k), (k, n) -> (m, n)`.

**Propagation Steps:**
1.  **Collect**: Input `A` says "My `k` factor is sharded on mesh axis `'model'`".
2.  **Resolve**: The system records: `Factor('k') = ['model']`.
3.  **Update**: Input `B` checks `Factor('k')`. It sees `['model']`. It applies this sharding to its own `k` dimension (dim 0).

This allows us to write rules once and handle any combination of sharding strategies (Data Parallelism, Tensor Parallelism, Sequence Parallelism) automatically.

---

## 4. Expectations for New Operations

When implementing a new `Operation`, you only need to define:

1.  **`maxpr`**: The local, single-device implementation (Standard MAX Graph API).
2.  **`sharding_rule`**: A template defining how factors map to inputs/outputs.
    *   *Simple Ops*: Use `elementwise_template`.
    *   *Complex Ops*: Define specific factor mappings (e.g., `(b, s, h, d) -> (b, s, h)`).
3.  **`infer_output_rank`**: Helper to determine rank stability.

The framework handles the rest: sharding inference, conflict resolution, communication injection (`AllGather`/`AllReduce`), and distributed execution.
