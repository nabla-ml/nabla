# Nabla SPMD Partial-Tensor Propagation — Architecture Review

Date: 2026-03-06

## Executive summary

The current partial propagation behavior is **mathematically on the right track** but **architecturally split across two encodings of the same concept**:

1. `ShardingSpec.partial_sum_axes` for axis-level deferred reductions with no surviving sharded output dimension.
2. `DimSpec.partial` for deferred reductions that remain attached to a sharded output dimension after contraction.

That split forces `spmd.py` into two separate reasoning paths and leaks complexity into op authoring, transform composition, and caching.

### Recommendation

Adopt a **revised internal architecture** centered on a single notion:

> a tensor may carry one or more **deferred reduction effects** over mesh axes.

Keep the existing public behavior and tests, but refactor the internals so every op sees and transforms the same effect object, regardless of whether the deferred reduction is currently “free” or “attached” to a specific sharded dimension.

This is **not** a recommendation for a big-bang rewrite. The safest path is:

- preserve current user-visible semantics,
- introduce a unified internal effect model,
- migrate `spmd.py`, `vmap`, `compile`, and autograd to it,
- then continue filling per-op coverage.

## Scope of this review

Files audited:

- `nabla/core/sharding/spec.py`
- `nabla/core/sharding/spmd.py`
- `nabla/core/sharding/propagation.py`
- `nabla/ops/base.py`
- `nabla/ops/binary.py`
- `nabla/ops/view/shape.py`
- `nabla/transforms/vmap.py`
- `nabla/transforms/compile.py`
- `nabla/core/autograd/backward.py`
- forward and transform tests under `tests/unit/`

Baseline verification:

- `tests/unit/test_stress_partial_propagation.py` passes: **14 passed**.

## What is already correct

The core mathematical policy is sound:

- if an op is distributive over the deferred reduction, the reduction may stay deferred,
- otherwise the reduction must occur before the op,
- and this must be decided independently per mesh axis.

The recent bug fixes fit that model cleanly:

- `CastOp`: narrowing casts are not distributive,
- `ConcatenateOp`: deferral requires all relevant operands to carry the same partial effect,
- `GatherOp` and `SliceTensorOp`: view-like selection ops are distributive.

The new stress suite is strong on forward numerical correctness because it uses:

- actual values, not trace strings,
- paired negative oracles,
- deterministic seeded inputs.

That testing standard should be treated as the template for all future partial-effect tests.

## Current architecture: where the complexity comes from

### 1. Same semantic fact, two encodings

Today a deferred reduction can appear in two forms.

#### A. Free partial effect

Stored in:

- `ShardingSpec.partial_sum_axes`

Meaning:

- the tensor is logically replicated in its visible dimensions,
- but each shard still contains only a partial contribution along one or more mesh axes,
- so a later `all_reduce` is required for materialization.

#### B. Attached partial effect

Stored in:

- `DimSpec.partial == True` on a dimension that is still sharded by some mesh axis.

Meaning:

- the output remains dimensionally sharded,
- and the values along that sharded dimension are also still partial because a contraction was deferred.

These are not fundamentally different phenomena. They are two placements of the same semantic object: a pending reduction effect.

### 2. The two-path split in `spmd.py`

Because of the dual encoding, `infer_output_sharding()` currently has two distinct decision modes:

- a “pure partial” fast path when there is no dimensional sharding but there are `partial_sum_axes`,
- a main factor-propagation path that produces `ghost_axes`, `reduce_axes`, and `DimSpec.partial`.

This is the clearest sign the abstraction boundary is wrong. The system is being asked to reason separately about:

- deferred reductions as axis metadata, and
- deferred reductions as a side effect of factor propagation.

But the compiler should only need one question:

> for each mesh axis, does this op preserve, consume, or require materialization of the deferred reduction effect?

### 3. Op authoring API is underspecified

Current op contract:

- `allows_partial_passthrough: bool`
- `partial_passthrough_axes(input_specs, kwargs) -> set[str]`

This has three problems.

#### Problem A: the boolean is mostly legacy noise

Once an op needs any real logic, the boolean no longer expresses the real contract.

Examples:

- `AddOp`: safe only if **all** operands are partial on the axis.
- `MulOp`: safe only if **at most one** operand is partial on the axis.
- `DivOp`: safe only when the numerator is partial and the denominator is not.
- `ConcatenateOp`: safe only if all concatenated operands are partial on the axis.

Those are not boolean properties of the op. They are **transfer rules over operand effect patterns**.

#### Problem B: return type is too weak

A returned `set[str]` only answers “which axes survive”. It cannot express:

- why an axis survives,
- whether the effect changed kind,
- whether it stayed attached to a specific output dimension,
- or whether multiple operand effects were merged.

#### Problem C: no semantic context

The method receives whole input specs, but the op has to infer everything ad hoc from low-level placement details.

## Key design finding

The current system is not wrong because it is conservative or permissive in the wrong places.
It is wrong as an architecture because it treats **placement** and **pending reduction** as partially separate worlds, then reunifies them with special-case code.

The right abstraction is:

- **placement** answers where data lives,
- **deferred reduction effects** answer which collectives are still semantically owed.

Then an op-specific rule transforms both.

## Proposed revised architecture

## 1. Introduce a unified deferred-reduction effect model

Internally, replace the split between `partial_sum_axes` and `DimSpec.partial` with a single per-axis structure.

Suggested shape:

```python
@dataclass(frozen=True)
class DeferredReduction:
    axis: str
    reduce_op: str          # "sum", later extensible to max/min/prod
    provenance: str         # "contraction", "scatter", "accumulate", "user"
    attached_dim: int | None
```

And then:

```python
@dataclass
class ShardingEffects:
    pending: dict[str, DeferredReduction]
```

Interpretation:

- `attached_dim is None`: current “free” partial effect.
- `attached_dim == k`: current `dim_specs[k].partial` case.

This unifies both existing representations without changing external behavior.

### Why provenance should exist

Today axis identity alone is used as the correctness key. That is enough for the current contraction-driven implementation, but it will not scale.

The legality of deferral can depend on why the value is partial:

- contraction sum,
- explicit scatter accumulation,
- future in-place accumulation semantics,
- custom collectives.

A view op may preserve all of them.
A nonlinear op may force all of them.
But other ops may care about the reduction kind or provenance.

Provenance should therefore be carried now, even if the initial enum has only one actively used value.

## 2. Replace the dual op API with one transfer function

Replace `allows_partial_passthrough` plus `partial_passthrough_axes()` with a single method.

Suggested direction:

```python
@dataclass(frozen=True)
class AxisEffectView:
    axis: str
    inputs_partial: tuple[bool, ...]
    inputs_attached_dim: tuple[int | None, ...]
    reduce_op: str
    provenance: tuple[str, ...]

@dataclass(frozen=True)
class DeferredReductionDecision:
    preserve: bool
    attached_output_dim: int | None = None
```

```python
def deferred_reduction_rule(
    self,
    axis_view: AxisEffectView,
    input_specs: list[ShardingSpec],
    kwargs: dict[str, Any] | None = None,
) -> DeferredReductionDecision:
    ...
```

That gives each op one answer per mesh axis:

- reduce before op,
- preserve as free pending reduction,
- preserve and attach to a specific output dimension.

### Why this is better

It makes common patterns explicit and composable:

- **view-like ops**: preserve unchanged,
- **all-input linear ops** like add/sub: preserve only if all operands carry the same effect,
- **single-sided linear ops** like mul/div by replicated value: preserve only for allowed operand patterns,
- **nonlinear ops**: always materialize,
- **contractions**: preserve but often reattach to an output dimension.

This turns partial propagation from ad hoc overrides into a real transfer system.

## 3. Collapse `spmd.py` to one effect pipeline

With unified effects, `infer_output_sharding()` can become one flow:

1. infer placement through factor propagation,
2. collect all pending deferred-reduction effects from inputs,
3. determine which effects survive the op via the op’s transfer rule,
4. decide whether each surviving effect is attached to an output dimension or remains free,
5. reduce any effect the rule rejects.

Under that model:

- the current fast path disappears,
- `ghost_axes` becomes unnecessary as a separate notion,
- `_save_multi_input_contracting_dims()` and `_restore_cleared_contracting_dims()` become transitional helpers or disappear entirely once factor propagation can emit attached deferred effects directly.

## 4. Keep placement and effects separate, but synchronized

Do **not** collapse everything into one overloaded `DimSpec`.
That recreates the current problem.

Instead:

- `DimSpec` remains about placement of tensor dimensions,
- effect state is stored separately and may reference an output dimension index.

This allows reasoning like:

- “axis `tp` shards output dim 1 and also carries a pending contraction-sum attached to dim 1”,
- or “axis `tp` is not visible in placement anymore but still owes a reduction”.

That distinction is real and should stay first-class.

## Assessment of the six open questions

## 1. Is the two-path split necessary?

**No.** It is an artifact of the current representation.

Both paths are implementing the same idea:

- some mesh-axis-local contributions are still pending reduction,
- and some ops may commute with that pending reduction.

A unified effect model removes the need for separate “fast path” and “ghost axis” logic.

## 2. Is the current op interface good enough?

**No.** The boolean + set API is serviceable for patching but not for scaling.

The architecture should move to a single transfer rule over axis effects.

Short version:

- keep a compatibility adapter temporarily,
- mark `allows_partial_passthrough` as legacy,
- migrate ops to the new rule one by one.

## 3. Should axis identity carry more semantics?

**Yes.** Add at least:

- `reduce_op`,
- `provenance`,
- optional attachment to output dimension.

This will matter for future correctness and for cost-based scheduling.

## 4. What about `vmap` and `batch_dims`?

This is currently the weakest compositional area.

### Observed concern

`nabla/transforms/vmap.py` constructs new sharding specs in `_apply_shard()` from `dim_specs` only. Existing `partial_sum_axes` and `replicated_axes` are not copied into the new spec.

That means batching logic is at risk of **dropping effect metadata** when it introduces or reassigns sharding on the vmapped axis.

Even if many current test patterns avoid partial outputs at that boundary, the abstraction is fragile.

### Required invariant

Batch transforms must preserve all pending reduction effects unless the transform itself semantically consumes them.

That needs to be an explicit invariant in the architecture, not an accident of cloning behavior.

## 5. What about `grad`, `vmap`, and `jit` composition?

Forward eager behavior is reasonably covered.
Transform-boundary behavior is not.

### Observed concern: autograd

Backward accumulation currently resolves partial cotangents mostly by matching only on axis names. This is safe, but it is eager and under-specified once multiple effect kinds exist.

### Observed concern: compile/jit cache identity

`nabla/transforms/compile.py` hashes only:

- mesh identity,
- `dim_specs.axes`,
- `dim_specs.partial`.

It does **not** include:

- `partial_sum_axes`,
- `replicated_axes`,
- future effect provenance.

That means two tensors with distinct pending-reduction states can alias to the same compile cache key.

This is a concrete architectural bug risk even if no failing test currently demonstrates it.

### Conclusion

Transform composition is not yet validated to the same standard as eager forward execution.
This should be the next testing priority after the internal effect model is cleaned up.

## 6. Should deferral become cost-model aware?

**Eventually yes, but not in the semantic layer.**

Correctness and optimization should be separated.

Recommended split:

- semantic layer computes the set of **legal** reduction schedules,
- optimization layer chooses among them.

Examples of future heuristics:

- reduce before narrowing cast,
- reduce after at most `N` linear ops to limit error growth,
- reduce earlier on bandwidth-rich meshes,
- preserve deferral through fused kernels when profitable.

But the first implementation should remain:

- correctness-first,
- math-safe,
- cost model optional.

## Concrete near-term issues worth fixing even before the full refactor

These are small, high-value follow-ups.

### A. `vmap` should preserve pending reduction metadata

When `_apply_shard()` creates a new sharding spec, it should preserve all existing effect metadata and only modify the intended batch placement.

### B. compile cache keys should include full effect state

At minimum, cache identity should include:

- `replicated_axes`,
- `partial_sum_axes`,
- and later the unified deferred-effect payload.

### C. effect-preserving helper APIs should exist in one place

Many operations need the same patterns:

- all inputs partial,
- at most one partial,
- lhs-only partial,
- pure view passthrough.

Those should become helper constructors for transfer rules rather than duplicated bespoke set logic.

## Recommended migration plan

## Phase 0 — keep current behavior green

- keep the 14-test stress suite as non-negotiable baseline,
- do not change user-facing sharding syntax yet.

## Phase 1 — introduce unified internal effect object

- add internal `DeferredReduction` representation,
- provide adapters to/from current `partial_sum_axes` and `DimSpec.partial`,
- keep existing public `ShardingSpec` fields during migration.

## Phase 2 — add new op transfer interface

- add `deferred_reduction_rule(...)`,
- provide default adapters for legacy ops using current boolean/set behavior,
- migrate representative ops first: add, mul, div, cast, concat, gather, slice, matmul.

## Phase 3 — simplify `spmd.py`

- remove the pure-partial fast path,
- replace `ghost_axes` bookkeeping with unified effect placement,
- delete restore/save hacks once contraction handling becomes native.

## Phase 4 — repair transform boundaries

- update `vmap` to transport effect state correctly,
- update compile cache keys,
- ensure autograd accumulation matches full effect identity, not only axis name.

## Phase 5 — expand rigorous tests

Add numerical + negative-oracle suites for:

1. `jit` / `compile` with partial outputs,
2. `grad` across deferred-reduction chains,
3. `vmap` over functions that return partial tensors,
4. nested `vmap` with `spmd_axis_name`,
5. batched row-parallel matmul followed by nonlinear ops,
6. transform compositions like `grad(vmap(f))`, `vmap(grad(f))`, and `jit(grad(f))`.

## Test strategy for the next round

Every new test about deferred reduction should follow the same pattern as the stress suite:

### Positive test

Compare Nabla output against unsharded JAX reference with `assert_allclose`.

### Negative oracle

Construct the wrong schedule explicitly.
For example:

- apply nonlinear op per partial shard, then sum,
- or batch/shard in a way that intentionally drops pending-reduction semantics.

Then assert the wrong result differs by a meaningful margin.

### Determinism

Use fixed unique seeds and explicitly assert the oracle difference is nontrivial.

## Priority recommendation

If only one architectural change is made next, it should be this:

> unify `partial_sum_axes` and `DimSpec.partial` behind one internal deferred-reduction effect model.

If only one testing expansion is made next, it should be this:

> add rigorous numerical tests for deferred reduction across `vmap`, `grad`, and `jit/compile` boundaries.

## Final conclusion

The current system is **behaviorally promising but structurally transitional**.

My recommendation is therefore:

- **do not** freeze the current architecture as final,
- **do** preserve its mathematical policy,
- **do** refactor around a unified deferred-reduction effect abstraction,
- **then** continue per-op expansion and cost-model work on top of that cleaner base.

In short:

- the **idea** is sound,
- the **representation** is not yet the right long-term one,
- and the biggest remaining risk is **transform-boundary correctness**, not forward eager per-op coverage.
