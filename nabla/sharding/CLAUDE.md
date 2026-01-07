# Nabla Sharding Architecture

This document serves as the authoritative reference for Nabla's distributed tensor system. It details the **Expected Behavior**, **Execution Pipeline**, and **Theoretical Foundations** of the sharding engine.

## 1. Core Philosophy

Nabla implements a **Unified SPMD (Single Program, Multiple Data)** architecture with **Factor-Based Propagation**.

### Key Principles
1.  **Uniformity**: Unsharded execution is just a special case (Mesh Size = 1). There are no separate "sharded" vs "unsharded" code paths in operations.
2.  **Symbolic Core, Concrete Shell**: sharding *propagation* is symbolic (factor-based), but *rule instantiation* handles concrete shape semantics (broadcasting, reshaping).
3.  **Lazy Evaluation**: We incur the cost of sharding inference only during graph building. Actual data movement happens only during execution/realization.
4.  **Expectation of Correctness**: Invalid sharding states (e.g., sharding a dimension of size 1) are caught during Propagation, ensuring runtime execution is always valid.

---

## 2. The Big Question: Why Shapes? Can't we just use Rank?

**Q: Do we really need specific input/output shapes, or is Rank (number of dimensions) enough?**

**A: Rank is NOT enough. Validity requires Shapes.**

While the *propagation algorithm* itself (`(m, k) -> (m)`) is symbolic, correctly **mapping** a tensor to that symbolic rule requires shape knowledge.

### Case 1: Implicit Broadcasting (The "Rank 2" Trap)
Consider `C = A + B`. Both are Rank 2.
*   **Scenario X**: `A:(10, 10)`, `B:(10, 10)`. Rule: `(d0, d1), (d0, d1) -> (d0, d1)`.
    *   Here, `A` and `B` share factors. Sharding `A` implies sharding `B`.
*   **Scenario Y**: `A:(1, 10)`, `B:(10, 10)`. Rule: `(new, d1), (d0, d1) -> (d0, d1)`.
    *   Here, `A`'s dim 0 is size 1. It CANNOT be sharded on a 4-device mesh. It is a "new" (or broadcast) factor.
    *   If we used only Rank, we would treat Scenario Y like Scenario X. We would try to shard `A`'s size-1 dimension, leading to crashes or silent data corruption.

### Case 2: Reshaping (The Conservation of Data)
Consider `Reshape(A) -> B`.
*   `A:(100, 20)`, `B:(2000,)`. Rule: `(a, b) -> (a * b)`.
    *   We map `A`'s dim 0 to factor `a` and dim 1 to `b`.
*   `A:(100, 20)`, `B:(40, 50)`. Rule: ???
    *   Without knowing the numbers, we don't know which input factors flow to which output factors. `100` splits into `40 * 2.5`? No.
    *   We utilize **Factor Sizing** to decompose dimensions: `100 = 2 * 50`, `20 = 20`. `40 = 2 * 20`.
    *   This decomposition requires concrete values to track how data moves.

**Conclusion**: The **Core Propagation** is symbolic (Factors), but the **Frontend (Rule Instantiation)** must be Shape-Aware to generate the *correct* symbolic rule.

---

## 3. Deep Dive: The Life of an Operation

What *exactly* happens when you call `z = x + y` (or any `Operation.__call__`)?

### Phase 1: The Setup (Frontend)
We are in Python. We have `Tensor` objects with metadata (`shape`, `sharding`, `mesh`).

1.  **Metadata Scan**: We check `x` and `y`.
    *   Are any sharded? Yes -> We extract the `DeviceMesh`.
    *   Are any traced? Yes -> We are building a graph.
2.  **Regularization**:
    *   If `x` is sharded but `y` is not, we temporarily assign `y` a `Replicated` sharding spec (empty axes). This brings everyone to the same "protocol".

### Phase 2: The Logic (Sharding Engine)
We need to decide: *How should the output `z` be sharded? Do inputs need to move?*

3.  **Rule Instantiation**:
    *   We look at `x.shape` and `y.shape`.
    *   We ask the Op: "Give me your logic." (e.g., `Elementwise`).
    *   The Op returns an `OpShardingRule`: e.g., `(d0, d1), (d0, d1) -> (d0, d1)`.
4.  **Propagation (The Solver)**:
    *   We feed the Rule + Input Specs into `propagate_sharding`.
    *   **Collect**: `x` says "I am sharded on `d0`". `y` says "I am replicated".
    *   **Resolve**: Factor `d0` is now marked as "Sharded".
    *   **Update**: We tell `y`: "You MUST be sharded on `d0` now". We tell `z`: "You inherits sharding on `d0`".
5.  **Output**:
    *   `output_spec`: Sharding for `z`.
    *   `input_specs`: Required sharding for `x` and `y`.
    *   `needs_allreduce`: Bool (Did we sum over a sharded dimension?).

### Phase 3: The Execution (Backend)
Now we have the Plan. We execute it.

6.  **Resharding (Alignment)**:
    *   We check `y`. Current: Replicated. Required: Sharded on `d0`.
    *   **Action**: `y = reshard(y, target=Sharded)`.
    *   *Note*: Ideally, `reshard` is smart. Moving Replicated->Sharded is just slicing. Moving Sharded->Replicated is AllGather.
7.  **SPMD Loop (The shard breakdown)**:
    *   We iterate over `device_id` from 0 to `N-1`.
    *   **Slice**: We get the *local piece* of `x` and `y` for this device.
        *   `x_local = x.get_shard(device_id)`
        *   `y_local = y.get_shard(device_id)`
    *   **Compute**: `z_local = maxpr(x_local, y_local)`. (This is the raw math op).
    *   **Collect**: We get a list `[z_0, z_1, ... z_N]`.
8.  **Reduction (If check failed in Phase 2)**:
    *   If `needs_allreduce` is True, we apply `AllReduce(sum)` across `z_local` chunks.
9.  **Packaging**:
    *   We wrap `[z_0, ...]` into a new `Tensor` object with the inferred `output_spec`.

---

## 4. Architectural Status & Future Directions

The core sharding infrastructure is **complete**:
- ✅ Factor-based propagation with 3-phase algorithm (Collect, Merge, Update)
- ✅ Unified SPMD dispatch (unsharded = mesh size 1)
- ✅ Complete communication ops: `ShardOp`, `AllGatherOp`, `AllReduceOp`, `ReduceScatterOp`, `GatherAllAxesOp`
- ✅ AllReduce insertion for sharded contracting dimensions (e.g., K in matmul)
- ✅ Lazy graph-based resharding (ops add to graph, not eager eval)

### Current Extension Points

### 1. Unified Broadcasting Logic
*   **Current State**: `BinaryOperation` manually handles broadcasting logic (unsqueezing/reshaping) in Python before calling `super().__call__`.
*   **Ideal State**: `Operation.sharding_rule` should handle broadcasting natively via **Multi-Input Broadcast Templates**. This would allow us to delete the manual broadcasting code in `binary.py`.

### 2. Autograd Integration
*   **Current State**: `vjp_rule` is separate.
*   **Ideal State**: Sharding propagation should automatically apply to the backward pass graph. Since the backward pass is just Ops, it *should* just work, but we need to verify `vjp_rule` definitions correctly propagate `kwargs` (like `axis`) which affect sharding.

### 3. VMap & Batching Integration

> [!IMPORTANT]
> This is a critical design area that requires careful thought before implementation.

#### How JAX Handles VMap + Sharding

JAX treats `vmap` and sharding as **orthogonal transformations** that compose:

1.  **`vmap` inside `jit(sharding)`**: Sharding happens *first*. Each device gets a shard. Then `vmap` vectorizes *within* that shard. The batch dimension `vmap` adds is **local** to each device.

2.  **`shard_map`**: Explicit SPMD. The function sees a *local shard*. If you want batching *within* that shard, you use `vmap` inside.

JAX's `PartitionSpec` applies to the **full array shape**, including any batch dimensions introduced by `vmap`. The XLA compiler then figures out how to map the batched, sharded computation efficiently.

#### The Core Tension: Who "Owns" the Dimension?

When you have a tensor of shape `(VMAP_DIM, REST...)`:
*   Is `VMAP_DIM` a "batch" dimension (managed by vmap)?
*   Or a "global" dimension (potentially sharded)?

**The order of transformation application matters!**
*   `jit(vmap(f))`: `vmap` runs first, creating a batched computation. Then `jit` compiles/shards the *batched* result. The batch dimension CAN be sharded.
*   `vmap(jit(f))`: `jit` compiles `f` for a single element. `vmap` maps this. Sharding inside `jit(f)` sees the *un-batched* shape.

#### Nabla's Current `batch_dims` Approach

`batch_dims` is stored on `TensorImpl`. This is similar to JAX's `vmap` internal state.
*   `LogicalAxisOperation` translates logical axes by adding `batch_dims` offset.
*   `Operation.sharding_rule` receives *logical* shapes (excluding batch dims).

**Current Behavior**: Sharding rules operate on the "inner" (non-vmapped) shape. This is consistent with `vmap(jit(f))`.

#### Concrete Scenarios

| Scenario | Input | VMap Axis | Sharding | Expected Behavior |
|---|---|---|---|---|
| **A** | `(4, 8)` | 0 (Batch) | None | Works trivially. No mesh. |
| **B** | `(4, 8)` | 0 (Batch) | Dim 1 sharded on `"d"` | Correct. `vmap` iterates Batch within each shard. Rule sees `(4,)` per shard. |
| **C** | `(4, 8)` | 0 (Batch) | **Dim 0** sharded on `"d"` | **Tricky!** Sharding the vmap axis. Each device sees `(2, 8)`. `vmap` should iterate the *local* 2. |

**Scenario C is the challenge.** If `batch_dims=1`, the `ShardingSpec` has `[DimSpec("d"), DimSpec([])]`. The first DimSpec covers the *physical* batch dimension.

#### Proposed Approach for Nabla

**Option 1: Current (Recommended for Now)**
*   Sharding applies to **logical** shape only.
*   `batch_dims` are implicitly replicated.
*   **Pro**: Simpler, covers common data parallelism use cases.
*   **Con**: Cannot shard the vmap batch dimension directly.

**Option 2: Full Unification (Future Work)**
*   `ShardingSpec` applies to **physical** shape (including batch dims).
*   `vmap` transform explicitly prepends `DimSpec([], is_open=True)` to `ShardingSpec`.
*   **Pro**: More powerful, allows sharding the vmap batch dimension.
*   **Con**: More complex, requires careful integration.

**Recommendation**: Start with Option 1. Users who need Scenario C can manually reshape/shard before vmapping.

---

## 5. Architectural Shortcomings & Known Limitations

> [!WARNING]
> These are fundamental architectural constraints, not bugs.

### 1. No Cost Model for Sharding Decisions

**Current Behavior**: `AGGRESSIVE` conflict resolution picks higher parallelism without considering communication cost.

**Impact**: May choose suboptimal shardings (e.g., sharding a dimension that requires expensive AllGather downstream).

**Mitigation**: Users can override with explicit `priority` on `DimSpec` annotations.

### 2. No Cross-Mesh Propagation

**Current Behavior**: All tensors in an operation must share the same `DeviceMesh`.

**Impact**: Cannot express pipelines that transition between different mesh topologies (e.g., DP mesh → TP mesh).

**Workaround**: Explicit gather-to-replicated, then re-shard on new mesh.

### 3. Single-Machine Simulation Only (Hardware Limitation)

**Current Behavior**: All shards run on the same device. `_is_distributed(mesh)` checks for unique device refs.

**Impact**: Real distributed issues (network latency, collective semantics, memory pressure) are untested.

**Status**: Awaiting MAX multi-device backend support.

### 4. Limited Sharding Template Coverage

**Current Templates**: `matmul`, `elementwise`, `reduce`, `transpose`, `broadcast`.

**Missing**: Attention, convolution, scatter/gather, custom user ops.

**Note**: This is intentional for rapid architectural iteration. Templates can be added incrementally.

### 5. Uneven Shard Handling

**Current Behavior**: `compute_local_shape` uses `math.ceil(dim_size / num_shards)`, creating uneven last shards.

**Impact**: Operations on shards with different sizes may require padding or masking (not automated).

### 6. No Automatic Gradient Sharding Verification

**Current Behavior**: VJP rules are separate from sharding rules.

**Risk**: A `vjp_rule` that doesn't correctly handle sharded kwargs (like `axis`) could produce incorrect gradients.

**Status**: Infrastructure ready; needs verification tests.
