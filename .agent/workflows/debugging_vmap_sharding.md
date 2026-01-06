# Vmap + Sharding Execution Model

This document outlines the "Mental Model" for how `vmap` and sharding (SPMD) interact in Nabla, following the "Strict Resharding" refactor.

## 1. The Vmap Protocol
`vmap` transforms a function `f(x) -> y` into `F(X) -> Y`.
*   **Input**: `_batch_tensor` moves `axis` to `batch_dim` (Physical 0).
*   **Sharding**: `shard_batch_dims` applies `ReshardOp` if `spmd_axis_name` is set.
*   **Ops**: Logical ops (View, Math) execute on "Local" slices, maintaining Global metadata.

## 2. Key Architecture: Lazy Graph, Strict Transitions

### The "Truth Gap" Resolution
A major challenge is ensuring operations running on local shards receive correctly sliced data, especially when `vmap` introduces implicit broadcasting (Global Data -> Sharded Context).

**Solution: Strict Resharding**
We strictly enforce that **Physical Data matches the Sharding Spec** at operation boundaries.

1.  **Infer Output Sharding**: Determines required specs for inputs.
2.  **Reshard Inputs**:
    *   Compares `Current Spec` vs `Target Spec`.
    *   **CRITICAL**: If `Target` is Sharded (e.g. `[DP]`) and `Current` is Replicated (e.g. `[R]` or `None`), we **INSERT** a `ReshardOp`.
    *   Optimization Removed: We no longer skip resharding for Replicated tensors.
    *   `ReshardOp` -> `ShardOp` -> **Physical Slicing**.
    *   Result: `x._impl.sharded` is True, and `_values` contains **Local Slices**.

3.  **Get Shard Args (Simplified)**:
    *   Since inputs are guaranteed to be sliced (if spec says so), `get_shard_args` is trivial.
    *   If `len(vals) > 1`: Multiple shards exist. Return `vals[shard_idx]`.
    *   If `len(vals) == 1`: Single value exists. Due to strict resharding, this **IS** the correct slice. Return it.
    *   No more guessing with `cached_shape`.

### 3. Case Study: `vmap(matmul(x, w))`
*   `x`: Sharded `[DP, K]`. Local `(B, K)`.
*   `w`: Unsharded `[K, N]`.
*   **Vmap Broadcasting**: `w` -> `Unsqueeze` -> `Broadcast` -> `(1, K, N)`. Spec `[R, R, R]`.
*   **Matmul Propagation**:
    *   Rule: `(b, k), (b, k, n) -> (b, n)`.
    *   `b` matches `DP`.
    *   `w` input required spec: `[DP, R, R]`.
*   **Resharding**:
    *   Current `[R, R, R]` vs Target `[DP, R, R]`.
    *   `needs_reshard` = True.
    *   `ReshardOp` inserted.
    *   `w` (Global) -> Sliced -> `w_resharded` (Local Slices).
*   **Execution**:
    *   `get_shard_args(w_resharded)` returns Local Slice `(1, K, N)`.
    *   `Matmul` executes `(B, K) @ (K, N)` (conceptually).
    *   Result is locally correct.

## 4. Debugging Tips
*   If `TypeError: maxpr missing args`: Check `reshard_inputs` logic handling empty/None mesh.
*   If Shape Mismatch `(16, 4) vs (4, 4)`: Strict Resharding failed to insert `ReshardOp`. Check `needs_reshard`.
