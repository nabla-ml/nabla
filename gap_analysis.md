# Sharding Implementation Assessment: Nabla vs. XLA Shardy

## Executive Summary

Nabla's sharding implementation has achieved a **high degree of fidelity** to the XLA Shardy specification in terms of **Representation** and **Local Propagation**. We have successfully implemented complex features like **Sub-axes**, **Priorities**, **Open/Closed Dimensions**, and **Compound Factors** (Reshape).

However, significant architectural gaps remain in the **Global Solver Strategy** and the **Execution Model**. Nabla currently uses a "Single-Pass" propagation and "Eager" execution, whereas XLA uses an "Iterative Priority Solver" and "Lazy/Partial" execution to enable advanced optimizations.

**Current Verdict:** A fully functional **Forward-Pass Sharding Simulator** that correctly handles complex layouts (including Resharding) but lacks the global optimization and fusion capabilities of a production-grade compiler.

---

## 1. Feature Fidelity Matrix

| Feature | Shardy Specification | Nabla Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Representation** | | | |
| Open/Closed Dims | `dim=?` (open) vs `dim=x` (closed). | `DimSpec.is_open`. | ✅ Full Parity |
| Sub-Axes | `x:(1)2` (split axes logic). | Logically supported in `spec.py` & `DeviceMesh`. | ✅ Full Parity |
| Priorities | `p0` (strong) ... `pN` (weak). | `DimSpec.priority`. | ✅ Full Parity |
| Explicit Replication | `replicated={"x"}` (constraint). | `ShardingSpec.replicated_axes`. | ✅ Full Parity |
| **Propagation** | | | |
| Algorithm | Factor-based (Collect -> Resolve -> Update). | Implemented in `propagation.py`. | ✅ Full Parity |
| Compound Factors | `(i,j) -> k` (Reshape/Merge/Split). | generic `reshape_template` & factor logic. | ✅ Full Parity |
| Conflict Resolution | Basic (Prefix) & Aggressive (Parallelism). | Implemented strategies. | ✅ Full Parity |
| **Solver Loop** | Iterative `for p in priorities: propagate`. | **Single-Pass** (Local greedy). | ⚠️ **Major Gap** |
| **Execution** | | | |
| Collective Ops | `AllReduce`, `AllGather`. | Simulated locally. | ✅ Generic |
| Resharding | Generic `Reshard` (AllToAll). | Implemented via `Gather->Slice`. | ✅ Generic |
| **Tensor State** | **Partial** (Pending Reduction). | **Missing** (Eager AllReduce). | ❌ **Blocking Fusion** |

---

## 2. Detailed Gap Analysis

### Gap A: The Missing "Iterative Priority Solver" (Algorithmic)
**Severity: Medium (Correctness in Edge Cases)**

XLA Shardy enforces priorities *globally* by running propagation in rounds:
1.  Solve all `p0` (highest priority) constraints globally until fixed point.
2.  Freeze `p0` decisions.
3.  Solve `p1`, and so on.

**Nabla's Approach:**
We run a **Single Pass** where priorities are checked locally at each step.
-   **Risk:** A low-priority constraint (`p1`) close to a tensor might "claim" a dimension before a high-priority constraint (`p0`) from a distant part of the graph propagates there.
-   **Impact:** In complex graphs with conflicting constraints, Nabla might settle on a sub-optimal or "incorrect" (w.r.t spec) sharding that respects local priorities but violates global precedence.

### Gap B: Eager Execution vs. Partial State (Performance)
**Severity: High (Optimization)**

In XLA/GSPMD, a generic Matrix Multiplication (`(m, k) @ (k, n)`) sharded on the contracting dimension `k` produces a tensor in a **Partial State**. This state conceptually holds local partial sums and *implies* a cleanup `AllReduce`, but does not execute it immediately.

**Nabla's Approach:**
Our `Operation._call_spmd` detects sharding on contracting factors and **immediately** injects/simulates an `AllReduce` before returning the result.
```python
# nabla/ops/operation.py
output, ... = self.maxpr(...) 
if reduce_axes:
    output = all_reduce(output, axes=reduce_axes)  # Eager!
return output
```
-   **Impact:** We cannot fuse reductions.
    -   *Example:* `Sum(MatMul(A, B))` should be a single `ReduceScatter` or `AllReduce`.
    -   *Nabla:* Executes `AllReduce` (for MatMul) -> `AllReduce` (for Sum), essentially double-counting communication or missing optimization opportunities.

### Gap C: Autograd Compatibility
**Severity: Unknown (Unverified)**

We have not yet traced or verified the backward pass for sharded operations.
-   **Challenge:** The backward pass of a "Simulated Sharded Op" must be carefully constructed. If we rely on the standard `autograd` engine seeing our "local simulation" (loops over shards), it might generate a valid but inefficient backward graph. Ideally, the backward pass of a "Sharded Op" should itself be a "Sharded Op" with derived sharding rules.

---

## 3. Path Forward

### Phase 7: Iterative Solver (Recommended Next Step)
Refactor `propagate_sharding` to wrap the unified pass in a priority loop.
-   Requires `ShardingSpec` to track "frozen" status for dimensions settled at higher priorities.

### Phase 8: Partial State & Fusion
Introduce a `Partial` state to `ShardingSpec` or `TensorImpl`.
-   Update `Operation` to return `Partial` tensors instead of running `AllReduce`.
-   Implement a "Realize" pass (or safe accessor) that inserts `AllReduce` only when a Partial tensor is consumed by an op that requires full data.

### Phase 9: Distributed Runtime
Replace the "Simulated" SPMD execution (local loop over shards) with actual multi-process execution using `torch.distributed` or similar, allowing real benchmarking.
