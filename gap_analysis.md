# Sharding Implementation Assessment: Nabla vs. Shardy (XLA)

## Executive Summary
**Current Status:** High-Fidelity Representation, simplified Propagation Driver.
**Verdict:** Nabla largely achieves the **Shardy Representation** fidelity (Open/Closed dims, Sub-axes, Priorities). However, its **Propagation Driver** is currently a simplified "single-shot" pass rather than the robust **Iterative Priority Solver** defined in Shardy.

Critically, the **Execution Model** diverges by lacking a **Partial Tensor State**, which prevents advanced optimizations (fusion of reductions) common in XLA/GSPMD.

---

## 1. Feature Fidelity: Nabla vs. Shardy Design

| Feature | Shardy Specification | Nabla Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Open/Closed Dims** | `dim=?` means open. | `DimSpec.is_open`. | ✅ Mapped |
| **Sub-Axes** | `x:(1)2` (split axes). | Logically supported in `spec.py` & `DeviceMesh`. | ✅ Mapped |
| **Explicit Replication**| `replicated={"x"}` (constraint). | `ShardingSpec.replicated_axes`. Filtered in Update. | ✅ Mapped |
| **Priorities** | `p0`...`pN`. Recursive solver loop. | Attributes exist on `DimSpec`, but **Solver Loop is missing**. | ⚠️ Partial |
| **Compound Factors** | `(i,j) -> k` (Reshape). | Not fully generic in `OpShardingRule` yet. | ⚠️ Partial |
| **Data Flow Ops** | `While`/`Case` Passthrough. | Not implemented. | ❌ Missing |

## 2. The Real Gaps

### A. Missing "Iterative Priority Solver"
**Severity: Medium**
Shardy defines a hierarchy: `User Priority > Op Priority > Aggressive > Basic`.
- **Shardy:** Solves `p0` constraints globally, then freezes them, then solves `p1`, etc.
- **Nabla:** Runs a single propagation pass. While it respects priorities locally, it doesn't strictly enforce the *global ordering* of decisions. This could lead to lower-priority constraints locally overriding higher-priority ones from distant nodes before the high-priority ones have propagated.

### B. Missing "Partial" Tensor State (Execution Gap)
**Severity: High**
In XLA/GSPMD, a generic `Dot` (MatMul) usually produces a tensor with a "Partial" layout (sharded on the contracting dimension). The reduction is *implied* but not executed immediately.
- **Why it matters:** This allows fusing multiple reductions (e.g. `BatchMatMul -> Sum` becomes `ReduceScatter` or single `AllReduce`).
- **Nabla:** We greedily injected `AllReduce` immediately in `ops/math.py`. We treat "Partial" as a transient state involved in a single Op, rather than a first-class Tensor state.

### C. General Resharding (AllToAll)
**Severity: Medium**
We support `AllGather` and `AllReduce`. We lack general `Reshard` logic (using `AllToAll`) that is required when axes are explicitly split or permuted (e.g. transpose interaction with split axes).

### D. Auto-Differentiation Compatibility
**Severity: Critical (Unknown)**
GSPMD automatically derives the sharding of the backward pass from the forward pass.
- **Risk:** We must ensure `autograd` can see through our "eager" communication ops or that the backward pass of a "Sharded Op" correctly invokes the necessary "Sharded Gradient Op" (e.g. `AllReduce` in fwd -> `AllGather` in bwd?).

## 3. Path Forward (Matching Shardy)

1.  **Implement the Solver Loop:** Update `infer_output_sharding` (or a global pass) to iterate `for p in priorities: propagate(max_p=p)`.
2.  **Introduce `Partial` Spec:** Allow `ShardingSpec` to denote "pending reduction on axis X". Update Ops to return this state instead of running `AllReduce`. Add a `Realize` pass to insert `AllReduce` before consumption.
3.  **Verify Gradient Flows:** Verify that `autograd` correctly decomposes distributed operations.

## Conclusion
Nabla is architecturally sound and surprisingly close to the XLA Shardy spec in terms of data structures. The gap lies in the **Algorithmic Complexity** of the solver (iterative vs one-shot) and the **Lazy Optimization** of reductions (Partial state).
