# Hessian Debugging Status Report - Physical Ops

## Overview
This document serves as a status report and handover for the debugging efforts regarding Hessian computations in Nabla, specifically focusing on physical operations and nested forward-mode autodiff (`fwd_fwd` and `rev_fwd`).

## Recent Progress
- **Refactored Base Hierarchy**: 
    - `BroadcastToPhysicalOp` now inherits from `ShapeOp`.
    - `PhysicalReduceOp` (sum, mean, max, min) now inherits from `AxisOp`.
    - `UnsqueezePhysicalOp` and `SqueezePhysicalOp` now inherit from `AxisOp`.
    - **Benefit**: These ops now leverage centralized `adapt_kwargs` to automatically handle `batch_dims` during physical execution.
- **Nested JVP Fix**: Resolved issues with nested tangent propagation. Tangent-parent relationships are now correctly preserved and cleared using `_clear_jvp_cache`.
- **Debug Infrastructure**: Integrated `NABLA_DEBUG_PHYS` flag across `base.py`, `shape.py`, and `axes.py` for tracing physical shape transformations.

## The Core Challenge: Batch Dimension Discrepancy
In higher-order differentiation (like `jacfwd` or `jacrev`), Nabla uses `vmap` internally to calculate derivatives with respect to basis vectors. This introduces a "Batch Dimension Discrepancy":

1. **The Primal Situation**: A primal operation (e.g., `broadcast_to_physical`) is called on a tensor with `batch_dims=B`.
2. **The AD Situation**: The `jvp_rule` or `vjp_rule` receives a tangent or cotangent that has been vmapped, so it has `batch_dims=B + E` (where `E` is the number of extra differentiation axes).
3. **The Failure**: The AD rule typically re-uses the `kwargs` (like `shape` or `axis`) from the primal call. However, these `kwargs` are relative to the primal's `batch_dims`. When applied to the tangent (which is rank-higher due to `E`), the physical operation fails because it doesn't account for the `E` extra dimensions at the front.

**Example**:
If a primal `broadcast_to_physical(x, shape=(2,3))` is called on `x` with `batch_dims=0`, and the JVP tangent has `batch_dims=1` (shape `[N, 2, 3]`), the JVP rule calling `broadcast_to_physical(tangent, shape=(2,3))` will fail because it tries to broadcast a rank-3 tensor to a rank-2 shape.

## Current Status (Failing Tests)
Target file: `tests/unit/test_hessian_physical_ops.py`
Status: **6 failures, 18 passed**.

### Key Failures & Blockers:
1. **`test_hessian_broadcast_to_physical`**:
   - **`fwd_fwd` Error**: Still hitting rank mismatches in the JVP rule.
   - **Diagnosis**: The `jvp_rule` needs to "lift" the `target_shape` by prepending the extra batch dimensions from the tangent.
2. **Implicit Physical Paths**:
   - `test_hessian_implicit_broadcast_batch_dims_chain` is failing with `broadcast_to` rank errors.
   - **Diagnosis**: Operations like `broadcast_batch_dims` (which uses `reshape` and `broadcast_to`) are sensitive to being nested. The `ShapeOp.adapt_kwargs` helper is prepending batch dims, but if it double-prepends or misses the "extra" ones, it fails.

## Strategic Guidelines for the Next Agent

### 1. Mathematical Tracing & Expected Behavior
- Trace the Nabla ops using `NABLA_DEBUG_OP_CALL=1`.
- A Hessian is a derivative of a derivative. If you see `reshape` or `broadcast_to` appearing in the trace, verify that their input/output shapes are exactly what you'd expect if you were manually differentiating the chain.

### 2. Negative Axis & KWarg Re-use
- **Strategy**: Leverage negative axes (`axis=-1`) wherever possible. They are naturally "batch-dim-prefix-agnostic".
- **AD Rules**: If a JVP/VJP rule uses a physical op, it **must** calculate the `extra_prefix` (e.g., `tangent.batch_dims - primal.batch_dims`) and adjust positive axes or prepend dimensions to the target shape.

### 3. Cleanup is Mandatory
- Always use `cleanup_caches()` between test runs. Stale JVP/VJP caches will cause "tangent already has a parent" or "tangent mismatch" errors that are unrelated to your current code changes.

### 4. Avoid "Manual Shifts" in physical kernels
- The `kernel` methods should be pure. All adaptation should happen in `__call__` (for logical-to-physical setup) or `adapt_kwargs` (for sharding/batching).
- Inheritance from `AxisOp` or `ShapeOp` is the preferred way to handle standard batch dimension propagation.

## Final Note
The goal is to make Nabla's physical operations as "transparent" to `vmap` as the logical ones. This requires a robust mechanism for physical ops to detect and respect the extra batch dimension prefix introduced by AD transforms.
