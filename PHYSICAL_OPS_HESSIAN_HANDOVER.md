# Physical Ops Hessian Debugging Handover

## ⚠️ IMPORTANT (READ FIRST)

Always debug with `Trace.__str__` output before changing rules. Do not guess from final tensor values alone.

For every failing case, inspect this exact ladder first:

1. `trace(f, x)`
2. `trace(jacfwd(f), x)` (or `jacrev(f)` depending on mode)
3. `trace(jacfwd(jacfwd(f)), x)` (or matching 2nd-order composition)

If traces are available, they are the ground truth for which Nabla ops are actually called and where signal is lost.

## Purpose

This document captures the current state of the physical-op Hessian debugging effort, so another engineer/agent can continue without repeating discovery work.

Scope: nested Jacobian/Hessian failures in Nabla physical ops (`rev_rev`, `fwd_fwd`, `rev_fwd`, `fwd_rev`), especially shape/batch prefix conflicts.

---

## Executive Summary

- This started as an op-level physical-shape problem, but current evidence says we have **two interacting fronts**:
  1) physical-op VJP/JVP shape/axis reconstruction bugs under nested prefixes,
  2) transform-level nested-forward (`fwd_fwd`) behavior still dropping second-order signal.
- Errors still often surface in `broadcast_to_physical`/`reshape`, but these are frequently **validators of upstream mistakes** (wrong axis/shape emitted by a caller rule).
- Several structural crash fronts are fixed, but correctness is not yet stable across all Hessian mode combinations.
- Use focused suites + structural probes first; do not trust broad suite noise until single-op invariants hold.

---

## 2026-02-18 Candid Status (Confidence vs Hacks)

### ✅ Fixed with high confidence

1. `EqualOp.jvp_rule` exists and returns zero tangent.
   - Effect: removes a hard blocker for higher-order paths touching `equal` masks (max/min derivatives).

2. Multiple hard shape crashes in focused physical reductions were removed.
   - Example classes that moved from crash → execution (at least in some modes): `reduce_sum_physical`, `mean_physical`.

3. Physical axis/squeeze handling is more robust than baseline.
   - Added defensive normalization and axis handling in physical reduction/view paths.

### ⚠️ “Hacky” or low-confidence changes (kept for progress, may be wrong globally)

1. Several physical view/reduction fixes were heuristic and case-driven.
   - They removed immediate shape exceptions but are not yet proven as global invariants.

2. Transform experiments in `jacfwd`/`jacrev` switched from vmap-basis to explicit basis loops.
   - This reduced some batch-prefix complexity but did **not** solve broad `fwd_fwd` zero-Hessian failures.
   - Treat as exploratory unless validated by targeted derivative-invariant tests.

3. Some batch-prefix compensation in physical broadcast/unsqueeze/squeeze paths is likely overfit.
   - It needs simplification once the true nested-forward bug is isolated.

### Current honest read

- We have made real progress on **structural crashes**.
- We have **not** resolved the global correctness issue for `fwd_fwd` Hessians.
- Some prior “core issue is only op-level” framing was too strong; transform-level nested tangent handling is still a credible primary culprit.

---

## Files Added / Modified During This Investigation

### Added

- `tests/unit/test_hessian_physical_ops.py`
  - Focused Hessian tests for explicit and implicit physical-op paths.
  - Covers all 4 Hessian constructions and includes mode-consistency tests for non-smooth reductions.

- `tmp_physical_trace.py`
  - Traces implicit op chains used during Jacobian/Hessian computation.

### Modified

- `nabla/ops/reduction.py`
  - Added physical reduction derivative rules:
    - `MeanPhysicalOp.vjp_rule`
    - `MeanPhysicalOp.jvp_rule`
    - `ReduceMaxPhysicalOp.vjp_rule`
    - `ReduceMaxPhysicalOp.jvp_rule`
    - `ReduceMinPhysicalOp.vjp_rule`
    - `ReduceMinPhysicalOp.jvp_rule`
  - Updated `ReduceSumPhysicalOp.vjp_rule` to physical broadcast path.
  - Added helper logic for physical-shape / axis normalization / extra-prefix handling.

- `nabla/ops/comparison.py`
  - Added `EqualOp.jvp_rule` returning zero tangent (`zeros_like(outputs[0])`) to unblock higher-order paths that touch `equal` masks.

---

## What We Learned (Important)

1. **Symptom location is not root cause location**
   - If failure says `broadcast_to_physical` rank mismatch, investigate the VJP/JVP rule that requested that broadcast target first.

2. **Nested AD introduces extra leading batch-like prefixes**
   - Rule logic that works for first-order AD can break for second-order because cotangents/tangents carry extra leading structure.

3. **Physical axis vs logical axis must stay explicit**
   - Physical ops should reason about `physical_global_shape` and `batch_dims` directly.
   - Avoid silently mixing logical assumptions into physical-rule code.

4. **Non-smooth ops (max/min) need execution-safe higher-order behavior**
   - We are not enforcing perfect math smoothness at tie points; we need robust and consistent execution paths across modes.

---

## Current Failure Profile (Latest Focused Run)

Command used:

```bash
./venv/bin/python -m pytest tests/unit/test_hessian_physical_ops.py -k "mean_physical or reduce_max_physical_modes_smoke or reduce_min_physical_modes_smoke or reduce_sum_physical" -q --tb=short
```

Observed:

- `reduce_sum_physical`:
  - `rev_rev`, `rev_fwd`: now execute but numeric mismatch remains.
  - `fwd_fwd`: reshape element-count mismatch remains.
  - `fwd_rev`: physical broadcast target mismatch remains.

- `mean_physical`:
  - mixed execution and numeric failures remain.
  - one failure showed invalid squeeze on non-1 dimension.

- `reduce_max_physical` / `reduce_min_physical`:
  - `equal` higher-order blocker was addressed by adding `EqualOp.jvp_rule`.
  - remaining failures are still shape/axis path issues in forward/reverse mixed modes.

---

## High-Confidence Debugging Heuristic

When a physical-op Hessian test fails:

1. Read the deepest failing op (`broadcast_to`, `reshape`, `squeeze`, etc.).
2. Do **not** patch that op first unless clearly wrong.
3. Walk one frame up to identify the VJP/JVP rule that invoked it.
4. Validate in that rule:
   - target physical shape construction,
   - axis normalization (`axis < 0` handling),
   - `keepdims` behavior,
   - extra-prefix handling when `tangent.batch_dims` / `cotangent.batch_dims` exceeds primal batch dims,
   - unsqueeze/squeeze axis selection under nested prefixes.
5. Re-run only the minimal failing test mode.

This catches root causes faster than patching generic view ops.

---

## Repro & Triage Workflow (Recommended)

### 1) Start with focused physical suite

```bash
./venv/bin/python -m pytest tests/unit/test_hessian_physical_ops.py -q --tb=short
```

### 2) Narrow to one op or mode

```bash
./venv/bin/python -m pytest tests/unit/test_hessian_physical_ops.py -k "reduce_sum_physical and fwd_rev" -vv --tb=long
```

For parametrized node selection in zsh, quote it:

```bash
./venv/bin/python -m pytest 'tests/unit/test_hessian_physical_ops.py::test_hessian_reduce_sum_physical[fwd_rev]' -vv --tb=long
```

### 3) Use implicit-op tracer

```bash
./venv/bin/python tmp_physical_trace.py
```

### 4) After each patch, re-run the smallest reproducer first

Only then run:

```bash
./venv/bin/python -m pytest tests/unit/test_hessian_physical_ops.py -q --tb=short
./venv/bin/python -m pytest tests/unit/test_hessian_four_combos_ops.py -q --tb=short
```

---

## Likely Remaining Root Causes

1. **Nested forward-mode state loss in transform stack (`fwd_fwd`)**
   - Symptom: many `fwd_fwd` Hessians are exactly zero where diagonal should be non-zero.
   - This pattern appears across unrelated ops (unary, binary, view, reduce), suggesting shared transform plumbing rather than isolated op formulas.

2. **Physical prefix ordering bugs in op rules still exist**
   - Known failures still include prefix-inverted broadcast targets and invalid reshape element counts in physical paths.

3. **Axis drift under mixed nested prefixes (`rev_*` and `*_rev`)**
   - Unsqueeze/squeeze/reduce reconstruction still occasionally applies primal-coordinate axes to cotangent/tangent tensors that carry extra leading dims.

4. **Potential gradient aggregation contamination in reverse paths for broadcasted binary chains**
   - Not fully proven, but matmul-related Hessian structure mismatches suggest some contributions may still be assembled with wrong leading semantics.

---

## Where To Look Next (Direct Answers)

### Is op creation partially at fault?

- **Possible but secondary.**
- The failing ops (`reshape`, `broadcast_to_physical`) are usually rejecting inconsistent metadata produced upstream.
- We have little evidence of kernel op-creation bugs themselves; evidence points to wrong arguments reaching valid kernels.

### Is the issue still in `std_basis` creation for `jacfwd`/`jacrev`?

- **Not ruled out, but not the strongest current signal.**
- `std_basis` shape construction may still interact badly with nested prefixes, but the dominant `fwd_fwd == 0` signature more strongly suggests nested tangent state handling / transform composition issues.
- Recommendation: verify with a tiny scalar sanity test that bypasses physical ops entirely.

### Can we point to op VJP/JVP rules as root cause by inspection?

- **Yes for specific physical shape crashes; no for global `fwd_fwd` zeros.**
- Op-rule inspection explains many physical broadcast/reshape exceptions.
- It does not fully explain why broad non-physical combos still lose second-order signal in forward-over-forward.

### Is binary op implicit broadcasting in `base.py` likely wrong?

- Current `BinaryOperation.__call__` does logical broadcast first, then batch-prefix equalization (`broadcast_batch_dims`), then physical broadcast.
- This ordering is conceptually correct.
- **Risk area:** when operands already carry nested prefixes, inserted implicit broadcasts can amplify any prior prefix misalignment from upstream transforms/op rules.
- Practical conclusion: keep it as suspect-adjacent, but not first suspect.

---

## Incremental Small-Test Strategy (Strict Ladder)

Use this exact order; advance only when current rung passes.

1. **Transform-only micro sanity (no physical ops)**
   - scalar/vector functions where Hessian is analytically obvious:
     - `sum(exp(x))` (diag = `exp(x)`),
     - `sum(x^3 + 2x^2 - 5x)` (diag = `6x + 4`).
   - Run all four mode combos; gate specifically on `fwd_fwd` being non-zero and correct.

2. **View-only logical chain**
   - `reshape -> moveaxis -> swap_axes -> polynomial reduce_sum`.
   - Confirms transform plumbing through view metadata without physical adaptation complexity.

3. **Physical single-op structural probes (`vjp`/`jvp`, not full Hessian)**
   - For each op family (`reduce_*_physical`, `broadcast_to_physical`, `unsqueeze/squeeze_physical`):
     - assert output tangent/cotangent logical shape,
     - assert `batch_dims`,
     - assert physical shape consistency.

4. **Single physical Hessian node tests (quoted node ids)**
   - one mode at a time, one op at a time.
   - prioritize currently flaky: `moveaxis_physical`, `broadcast_to_physical`, `unsqueeze_squeeze_physical`.

5. **Focused file suites**
   - `tests/unit/test_hessian_physical_ops.py` first,
   - then `tests/unit/test_hessian_four_combos_ops.py`.

6. **Only after all above: broader suite**
   - avoid broad runs while rung 1–4 still fail; they create noise and hide regressions.

---

## TODO / Timeline (Updated)

### Done

- Added physical Hessian suite and structural probe script.
- Fixed `equal` higher-order blocker.
- Removed multiple physical crash fronts (shape exceptions) in focused paths.

### In progress

- Separate transform-level `fwd_fwd` signal-loss bug from physical-op reconstruction bugs.
- Determine whether recent jacfwd/jacrev loop-based edits are real fixes or dead-end complexity.

### Next (ordered)

1. Add transform-only micro tests (rung 1) as hard gate.
2. Instrument nested `jvp` tangent-slot lifecycle (attach/restore) with assertions.
3. Re-validate `jacfwd`/`jacrev` basis assembly against those gates.
4. Only then simplify physical-op heuristics into invariant-based rules.
5. Re-test implicit broadcast interactions in binary op chains after transform gates pass.

---

## Suggestions for Next Engineer

1. Add temporary assertions in physical reduction rules:
   - print/assert `x_phys`, `cot_phys`, `tangent_phys`, `batch_dims`, computed axis, computed target shape.
2. Validate helper assumptions with one known failing case (`reduce_sum_physical[fwd_rev]`).
3. Treat `broadcast_to_physical` and `reshape` as **validators** of upstream rule math.
4. Keep fixes surgical; avoid broad transform rewrites unless op-level invariants are proven correct first.
5. Once stable, add regression tests for each fixed pattern (especially mixed-mode + nonzero prefix case).

---

## Practical Invariants to Enforce in Physical VJP/JVP Rules

- If `keepdims=False`, the cotangent/tangent rank differs by exactly one reduced axis (before considering extra prefixes).
- Any extra nested-AD prefix should be preserved consistently across all operands in the rule.
- Unsqueeze/squeeze axis should be computed in the actual tensor coordinate space being transformed (including prefixes).
- Broadcast targets should be built from real physical shapes, not inferred logical shortcuts.

---

## Status

- Investigation is in active fix phase.
- Major discovery work is complete.
- Remaining work is targeted rule-correctness on physical reduction/view interactions under nested prefixes.

---

## Latest Findings (Current Session Delta)

### What changed in this session

- Added `EqualOp.jvp_rule` in `nabla/ops/comparison.py` to unblock higher-order paths through `equal` masks.
- Iterated heavily on `nabla/ops/reduction.py` physical reduction rules (`reduce_sum_physical`, `mean_physical`, `reduce_max_physical`, `reduce_min_physical`).
- Added/removed variants of axis handling and cotangent broadcast target construction to isolate root behavior.

### Most informative trace outcome

Using `NABLA_DEBUG_OP_CALL=1` on

```bash
./venv/bin/python -m pytest 'tests/unit/test_hessian_physical_ops.py::test_hessian_reduce_sum_physical[rev_rev]' -q --tb=short
```

showed that nested paths can call `reduce_sum_physical` with tensors carrying nontrivial `batch_dims`, and axis adaptation consistency is critical.

Observed failure frontier currently includes:

- `ValueError: Cannot broadcast shapes (2, 2, 4) and (2, 3, 4)`
- (earlier variant during iteration): `Cannot broadcast shapes (24,) and (2,)`

Interpretation:

- Some cotangent contributions to the same primal are reconstructed with inconsistent logical shapes in reverse-over-reverse flow.
- This is still consistent with the main thesis: downstream broadcast/add failures are consequences of upstream VJP/JVP axis/shape reconstruction mismatch.

### Updated narrowed suspects

1. `ReduceSumPhysicalOp.vjp_rule` (and by analogy `MeanPhysicalOp.vjp_rule`) under nested prefixes.
2. Interaction between physical reduction axis adaptation and squeeze/unsqueeze reconstruction when `batch_dims > 0` in nested AD.
3. Potential secondary coupling with batch-dim metadata transitions (`incr_batch_dims`/`decr_batch_dims`) after reduction reconstruction.

### Practical next step (highest ROI)

For one failing case (`reduce_sum_physical[rev_rev]`), instrument `ReduceSumPhysicalOp.vjp_rule` with temporary prints/asserts for:

- `x.shape`, `x.batch_dims`, `x.physical_global_shape`
- `cot.shape`, `cot.batch_dims`, `cot.physical_global_shape`
- chosen unsqueeze axis, target logical shape, target physical shape

Then verify that every returned cotangent for the same primal has identical logical shape and compatible batch prefix before accumulation.

---

## Latest Root-Cause Progress (New)

### Confirmed upstream culprit pattern

Using targeted cotangent-flow tracing, we confirmed the final failing `add` is not root cause; it receives incompatible cotangents produced earlier by physical/batch ops.

Observed conflicting cotangents for same primal `(2, 3, 4)`:

- existing: logical `(1, 2, 4)`, `batch_dims=1`, phys `(24, 1, 2, 4)`
- incoming: logical `(2, 3, 4)`, `batch_dims=1`, phys `(24, 2, 3, 4)`

### Fix already applied

- `nabla/ops/view/batch.py` (`BroadcastBatchDimsOp.vjp_rule`):
   - Replaced direct `reduce_sum_physical(... axis=extra_prefix ...)` loop with:
      1. `move_axis_from_batch_dims(... batch_axis=extra_prefix, logical_destination=0)`
      2. `reduce_sum(... axis=0, keepdims=False)`
   - This avoids physical-axis misinterpretation under nested axis adaptation.

Impact:

- `test_hessian_reduce_sum_physical[rev_rev]` moved from shape-crash to numeric mismatch (progress: structural crash removed in this path).

### Remaining frontier

- `reduce_sum_physical` modes still fail numerically (`rev_rev`, `fwd_fwd`, `rev_fwd`), and `fwd_rev` still has a physical broadcast target mismatch.
- Structural probe shows suspicious JVP outputs under extra tangent batch dims (e.g. `reduce_sum_physical` JVP tangent shape `(1, 4)` where `(2, 4)` is expected).

---

## Structural Probe (Beyond Trial-and-Error)

Added diagnostic script:

- `tmp_experiments/_tmp_physical_vjp_jvp_structural.py`

Purpose:

- Runs single `vjp`/`jvp` probes per physical op family with cotangents/tangents carrying extra `batch_dims` vs primals.
- Prints logical shape, `batch_dims`, and physical shape for outputs and derivatives.

Run:

```bash
./venv/bin/python tmp_experiments/_tmp_physical_vjp_jvp_structural.py
```

This gives a fast, deterministic map of where physical-op derivative metadata is inconsistent before nested Hessian composition amplifies it.

---

## Debug Flags Added (Temporary, Opt-in)

For deep tracing during active debugging:

- `NABLA_DEBUG_PHYS_VJP=1` (in `nabla/ops/reduction.py`)
- `NABLA_DEBUG_COT_ACCUM=1` (in `nabla/core/autograd/utils.py`)
- `NABLA_DEBUG_COT_FLOW=1` (in `nabla/core/autograd/utils.py`)

Use only for diagnosis; disable by default for normal runs.

---

## New Trace-Math Diagnosis (2026-02-18, later session)

### Why this was not ad-hoc

We compared full `Trace.__str__` output for the same scalar test function

- `f(x) = reduce_sum(exp(x))`

across all Hessian compositions and matched each trace against expected second-order math.

### Critical trace mismatch that explains `fwd_fwd` failures

Before fix, trace of `jacfwd(jacfwd(f))` contained only basis assembly/view ops:

- `zeros -> unsqueeze -> concatenate -> reshape -> permute`

and **did not include any primal derivative path ops** (`exp`, `mul`, etc.).

Mathematically, for this `f`, Hessian must be diagonal `diag(exp(x))`; if trace has no dependence on `x`, outer forward pass sees a constant graph and emits zero Hessian.

This exactly matched observed failures (`fwd_fwd` all zeros in `test_hessian_four_combos_ops.py`).

### What was tried and rejected

- Attempted nested-`jvp` tangent chaining in `nabla/transforms/jvp.py`.
- Result: restored nonzero signal but wrong structure (dense repeated rows instead of diagonal).
- Interpretation: one-slot tangent representation cannot safely encode full nested forward semantics yet.
- Patch was reverted.

### Fix applied (safe, transform-level)

File changed:

- `nabla/transforms/jacfwd.py`

Behavior:

- Detect nested `jacfwd` composition (`jacfwd(jacfwd(...))`) via function marker.
- Route outer `jacfwd` through `jacrev` over inner `jacfwd` (`jacrev_over_jacfwd`) as a temporary correctness fallback.
- Mark wrappers with `_nabla_transform = "jacfwd"` to keep detection explicit.

Rationale:

- This preserves correct Hessian math while full multi-level forward-mode tangent tagging is not implemented.
- Avoids further shape/axis churn inside physical-op rules from a transform-level zero-signal bug.

### Validation results

Commands and outcomes:

```bash
./venv/bin/python -m pytest -q 'tests/unit/test_hessian_four_combos_ops.py::test_hessian_combo_unary_exp_sum[fwd_fwd]' --tb=short
# PASS

./venv/bin/python -m pytest tests/unit/test_hessian_four_combos_ops.py -k fwd_fwd -q --tb=short
# 7 passed

./venv/bin/python -m pytest tests/unit/test_hessian_four_combos_ops.py -q --tb=short
# 28 passed
```

Post-fix trace of `jacfwd(jacfwd(f))` now includes expected derivative path ops (`exp`, `mul`, `add`, basis slices), no longer just zero/basis scaffolding.

### Important correction (later same-day)

The temporary `jacfwd(jacfwd) -> jacrev(jacfwd)` fallback is **not accepted as final design** and was reverted for continued root-cause work.

Current effort is back on true nested-forward behavior (`jacfwd(jacfwd)` via nested `jvp`) using trace strings + micro probes.

### Remaining issue after transform fix

`tests/unit/test_hessian_physical_ops.py -k fwd_fwd` now has a narrowed remaining failure:

- `test_hessian_broadcast_to_physical[fwd_fwd]`
- reshape element-count mismatch during graph replay (`(4,1) -> (4,1,1,3)`)

Interpretation:

- Transform-level zero-Hessian bug is addressed for generic combos.
- Remaining failure frontier is now truly physical-op reconstruction/shape handling (especially `broadcast_to_physical` + view replay path).

---

## Latest Nested-JVP Root Findings (Trace-Driven, no fallback)

### 3-trace ladder result (as requested)

For `f(x)=reduce_sum(x*x*x)` at scalar/length-1 input:

1. `trace(f)` is correct: `mul -> mul -> reduce_sum -> squeeze`.
2. `trace(jacfwd(f))` is also structurally correct and includes expected product-rule terms.
3. `trace(jacfwd(jacfwd(f)))` is **not fully correct numerically** despite nonzero graph:
   - it includes derivative ops, but resulting coefficient is wrong (`10` instead of `12` at `x=2`).

So this is no longer a pure "all-zero" failure; it is now a **missing-term/level-loss** failure in nested forward propagation.

### Concrete scalar sanity numbers

At `x=2`, with unit nested directions:

- Expected for `y=x^3`: `y=8`, `dy=12`, `d2y=12`.
- Observed during probes: `d2y` can appear as `8` or `10` depending on path/context.

This confirms a product-rule contribution is dropped in some nested paths.

### What currently explains this behavior

Nested `jvp` is possible in current architecture, but recursion suppression is coarse:

- In `nabla/ops/base.py`, `Operation.__call__` triggers `apply_jvp(...)` when any input has tangent.
- In `nabla/ops/utils.py`, `apply_jvp` avoids infinite recursion by clearing input tangents before calling `op.jvp_rule`.

This mechanism prevents same-level infinite recursion, but with only one tangent slot it can also erase or mis-thread outer-level tangent information for intermediates.

### Why infinite recursion does not happen today

Because `apply_jvp` strips tangents from primals before running `jvp_rule`, ops executed *inside* `jvp_rule` typically do not re-enter the same-level tangent propagation endlessly.

So: recursion is controlled, but nested-level correctness is not yet guaranteed.

### Current status (honest)

- ✅ zero-Hessian collapse for some forward-over-forward paths was reduced.
- ✅ unary exp scalar case can now produce nonzero second-order signal in focused check.
- ❌ general `fwd_fwd` correctness remains broken.
- ❌ nested tangent-level handling is still over-complex and not principled (single-slot tangent emulation is fragile).

### Exact currently failing `fwd_fwd` tests (latest run)

From:

```bash
./venv/bin/python -m pytest tests/unit/test_hessian_four_combos_ops.py -k fwd_fwd -q --tb=short
```

Result: `.FFFFFF` (1 pass, 6 fail)

Passing:

- `test_hessian_combo_unary_exp_sum[fwd_fwd]`

Failing:

- `test_hessian_combo_binary_poly[fwd_fwd]`
- `test_hessian_combo_matmul_quadratic[fwd_fwd]`
- `test_hessian_combo_reduce_axis_chain[fwd_fwd]`
- `test_hessian_combo_view_chain[fwd_fwd]`
- `test_hessian_combo_broadcast_chain[fwd_fwd]`
- `test_hessian_combo_getitem_scalar[fwd_fwd]`

Observed patterns from this run:

- some cases remain all-zero Hessians (`matmul_quadratic`, `getitem_scalar`),
- others are nonzero but wrong coefficients/signs (`binary_poly`, `reduce_axis_chain`, `broadcast_chain`),
- view chain still has missing diagonal structure.

### Most important concrete probe (nested product rule still wrong)

Micro probe with manual nested tangent attachment at `x=2` for `b=(x*x)*x`:

- `x.tangent = (1, None)`
- `a = x*x`, `a.tangent = (4, 2)`  (already wrong: expected second level `2`, first level should be `4`; this mixed form indicates level confusion)
- `b = a*x`, `b.tangent = (12, 10)`

Expected for `x^3` with unit nested directions:

- primal `8`, first tangent `12`, second tangent `12`

Observed:

- second tangent `10` (or in other paths `8`)

Interpretation:

- a product-rule contribution is still dropped/mis-threaded across nested forward levels in multiplication chains.

### Next engineering step (recommended, high confidence)

Implement explicit forward trace levels (stack/tagged tangents) instead of ad-hoc parent fallback:

1. represent tangent as per-level mapping/stack (not single mutable slot),
2. in `apply_jvp`, pop only current level while preserving outer levels,
3. make `jvp_rule` evaluation operate at one explicit level,
4. keep trace-based micro gates (`x^2`, `x^3`, `sum(exp(x))`) as mandatory correctness checks before broader tests.

---

## ⚠️ IMPORTANT (READ LAST)

Do not continue by editing op rules blindly.

Before any new fix, print and compare the 3 trace levels (`f`, first derivative, second derivative) for the same failing input.

Use trace repr as the primary debugger for this issue: it is the fastest way to see exactly where Nabla stops propagating higher-order signal.
