# Nabla SPMD Sharding / Partial-Propagation — Handover

Date: 2026-03-06

## Purpose

This handover is for the next developer or AI agent continuing work on Nabla's SPMD sharding, deferred-reduction propagation, and transform composition.

It should be read together with:

- [docs/dev/spmd_partial_architecture_review.md](docs/dev/spmd_partial_architecture_review.md)

That review explains the architectural concerns and long-term direction.
This handover focuses on:

- what was already done,
- what was refactored in this session,
- what is still risky or incomplete,
- and what to do next.

---

## High-level state

Nabla's partial propagation system is working materially better than before.

### Confirmed-good areas

- forward numerical partial propagation stress tests,
- key sharding-transform interactions,
- compile cache identity for sharded/effectful inputs,
- `all_reduce(sum)` JVP support,
- removal of sharding-related `xfail`s in the focused transform suite.

### Still true architecturally

The implementation is still transitional.
The main architectural concern remains:

- the system represents deferred reductions in **two forms**:
  - free axis effects via `ShardingSpec.partial_sum_axes`
  - attached effects via `DimSpec.partial`

That should still be unified later.

---

## Original background

The system started from a working but ad hoc partial propagation model:

- row-parallel / contraction-style matmuls can yield partial sums,
- linear/distributive ops may defer the `all_reduce`,
- nonlinear ops must force reduction before applying the op.

Before this session, the following correctness fixes were already in place:

- `CastOp` no longer incorrectly defers narrowing casts,
- `ConcatenateOp` no longer incorrectly defers when only some inputs are partial,
- `GatherOp` correctly defers,
- `SliceTensorOp` correctly defers.

The rigorous forward test baseline was already present in:

- [tests/unit/test_stress_partial_propagation.py](tests/unit/test_stress_partial_propagation.py)

---

## What was added/refactored in this session

## 1. Sharding/effect utilities were made more explicit

In:

- [nabla/core/sharding/spec.py](nabla/core/sharding/spec.py)

added helpers:

- `ordered_axes()`
- `all_partial_axes()`
- `effect_signature()`

### Why

These are small but useful building blocks:

- better readability,
- stable ordering for display/cache identity,
- one place to ask for the complete deferred-reduction effect set,
- one stable signature for transform/cache logic.

---

## 2. Partial passthrough logic was de-duplicated

In:

- [nabla/ops/base.py](nabla/ops/base.py)
- [nabla/ops/binary.py](nabla/ops/binary.py)
- [nabla/ops/view/shape.py](nabla/ops/view/shape.py)

added shared helpers for common partial-passthrough patterns:

- preserve if all inputs carry the axis,
- preserve if at most `n` inputs carry the axis,
- generic axis-local predicate filtering.

### Why

Per-op logic was starting to duplicate the same patterns in slightly different forms.
This makes the current API less messy while the bigger architectural rewrite is still pending.

---

## 3. Shard transitions now preserve deferred-reduction metadata explicitly

In:

- [nabla/ops/communication/shard.py](nabla/ops/communication/shard.py)

`shard()` now accepts and transports:

- `partial_sum_axes`

and relevant internal paths were updated so reshard/shard transitions do not silently drop those effects.

### Why

Transform and communication boundaries were too easy a place to lose metadata.
This was one of the practical fragility points called out in the architecture review.

---

## 4. `vmap` SPMD batching was hardened

In:

- [nabla/transforms/vmap.py](nabla/transforms/vmap.py)

`_apply_shard()` now copies forward source sharding metadata when adding a batch sharding axis.

### Why

The prior logic rebuilt sharding mostly from `dim_specs`, which made effect metadata fragile.
This is still not the final abstraction, but it is safer and more explicit.

---

## 5. Compile cache identity was fixed

In:

- [nabla/transforms/compile.py](nabla/transforms/compile.py)

changes:

- compile now realizes lazy tensors before both cache-hit and cache-miss execution paths,
- sharding cache identity now uses `ShardingSpec.effect_signature()` instead of a weaker partial view.

### Why

Before this change, compile cache identity could ignore important sharding/effect metadata.
That was a real correctness risk.

---

## 6. `all_reduce(sum)` JVP was implemented

In:

- [nabla/ops/communication/all_reduce.py](nabla/ops/communication/all_reduce.py)

added:

- `AllReduceOp.jvp_rule()`
- `PMeanOp.jvp_rule()`

Also added a crucial guard:

- when `all_reduce` is only materializing an already-deferred partial effect from the primal path, the tangent must **not** be reduced again, or it double-counts.

### Why

This removed focused sharding-transform `xfail`s and made forward-mode AD materially more robust in sharded paths.

---

## 7. Focused tests were improved, and `xfail`s removed where actually fixed

Updated or added tests in:

- [tests/unit/test_transforms_sharded.py](tests/unit/test_transforms_sharded.py)
- [tests/unit/test_communication_rigorous.py](tests/unit/test_communication_rigorous.py)
- [tests/unit/test_vmap_sharding.py](tests/unit/test_vmap_sharding.py)
- [tests/unit/test_compile.py](tests/unit/test_compile.py)

Notably:

- removed sharding-related `xfail`s for JVP cases that were actually fixed,
- added direct regression coverage for compile sharding cache identity,
- kept the stress partial-propagation suite as the baseline numerical oracle.

---

## Targeted test commands that were run successfully

Use these exact focused commands rather than broad full-suite runs.

### Core partial propagation

```bash
venv/bin/python -m pytest -q tests/unit/test_stress_partial_propagation.py
```

### Transform + partial/sharding interactions

```bash
venv/bin/python -m pytest -q tests/unit/test_transforms_sharded.py
```

### Communication rigor

```bash
venv/bin/python -m pytest -q tests/unit/test_communication_rigorous.py
```

### Vmap + sharding + compile regression coverage

```bash
venv/bin/python -m pytest -q tests/unit/test_vmap_sharding.py tests/unit/test_compile.py tests/unit/test_communication_rigorous.py
```

### Broader focused bundle used during this session

```bash
venv/bin/python -m pytest -q \
  tests/unit/test_stress_partial_propagation.py \
  tests/unit/test_vmap_sharding.py \
  tests/unit/test_compile.py \
  tests/unit/test_transforms_sharded.py \
  tests/unit/test_communication_rigorous.py
```

All of the above passed at the end of this session.

---

## Files that matter most now

### Core implementation

- [nabla/core/sharding/spec.py](nabla/core/sharding/spec.py)
- [nabla/core/sharding/spmd.py](nabla/core/sharding/spmd.py)
- [nabla/core/sharding/propagation.py](nabla/core/sharding/propagation.py)
- [nabla/ops/base.py](nabla/ops/base.py)
- [nabla/ops/binary.py](nabla/ops/binary.py)
- [nabla/ops/view/shape.py](nabla/ops/view/shape.py)
- [nabla/ops/communication/shard.py](nabla/ops/communication/shard.py)
- [nabla/ops/communication/all_reduce.py](nabla/ops/communication/all_reduce.py)
- [nabla/transforms/vmap.py](nabla/transforms/vmap.py)
- [nabla/transforms/compile.py](nabla/transforms/compile.py)
- [nabla/core/autograd/backward.py](nabla/core/autograd/backward.py)
- [nabla/core/autograd/forward.py](nabla/core/autograd/forward.py)

### Tests

- [tests/unit/test_stress_partial_propagation.py](tests/unit/test_stress_partial_propagation.py)
- [tests/unit/test_transforms_sharded.py](tests/unit/test_transforms_sharded.py)
- [tests/unit/test_communication_rigorous.py](tests/unit/test_communication_rigorous.py)
- [tests/unit/test_vmap_sharding.py](tests/unit/test_vmap_sharding.py)
- [tests/unit/test_compile.py](tests/unit/test_compile.py)
- [tests/unit/common.py](tests/unit/common.py)

---

## What is still incomplete / risky

## 1. The core architectural split is still there

This is still the main open issue.

The system still encodes deferred reduction effects in two places:

- `partial_sum_axes`
- `DimSpec.partial`

That still causes complexity in:

- `spmd.py`
- transform composition
- communication/output-spec logic
- test reasoning

### Recommendation

Do the next real architectural step:

- introduce one internal deferred-reduction effect object,
- adapt current sharding structs to it,
- then simplify `spmd.py` around that single model.

This remains the highest-value next refactor.

---

## 2. `spmd.py` still has split reasoning paths and contraction-era bookkeeping

Read carefully:

- [nabla/core/sharding/spmd.py](nabla/core/sharding/spmd.py)

Still-important complexity points:

- pure-partial fast path,
- main factor-propagation path,
- `_save_multi_input_contracting_dims()`
- `_restore_cleared_contracting_dims()`
- `ghost_axes`

These are signs of representation mismatch.
They are not necessarily wrong, but they are still not the final architecture.

---

## 3. Transform-boundary semantics need more rigorous numerical coverage

We improved this area, but it is still under-tested relative to the forward partial suite.

### Especially worth expanding

- `grad(vmap(f))` where `f` produces/consumes partial tensors,
- `vmap(grad(f))` with sharded contraction paths,
- `jit/compile` over functions whose inputs or intermediates carry deferred reductions,
- nested transforms where batching and deferred reduction both interact.

### Important testing rule

Follow the same standard used in the original stress suite:

1. numerical equality against JAX/unsharded reference,
2. negative oracle proving sensitivity,
3. deterministic seeds.

---

## 4. There may still be sharding-adjacent `xfail`s elsewhere

In this session, the sharding JVP `xfail`s in:

- [tests/unit/test_transforms_sharded.py](tests/unit/test_transforms_sharded.py)

were fixed properly and removed.

But there are still other `xfail`s elsewhere in the repository that were not addressed here.
Those should be inspected case by case rather than normalized away.

---

## Suggested next tasks, in order

## Task 1 — unify deferred-reduction representation

Goal:

- replace the practical split between free and attached partial effects with one internal effect model.

Start from:

- [docs/dev/spmd_partial_architecture_review.md](docs/dev/spmd_partial_architecture_review.md)

Do this incrementally, not as a big-bang rewrite.

---

## Task 2 — create rigorous transform-boundary numerical stress tests

Create a new focused test file for something like:

- `tests/unit/test_partial_transform_composition.py`

Suggested contents:

- `grad` over row-parallel matmul followed by deferred linear chains,
- `vmap` over functions that take partial tensors and return nonlinear results,
- `compile` on functions whose inputs differ only in deferred-reduction metadata,
- nested transform cases.

All with JAX reference + negative oracles where appropriate.

---

## Task 3 — simplify `spmd.py`

After introducing a unified effect model:

- remove the pure-partial special path,
- eliminate `ghost_axes` as a separate concept if possible,
- fold the contraction save/restore logic into first-class effect propagation.

---

## Task 4 — audit communication ops for full forward-mode support

Now that `all_reduce(sum)` and `pmean` have JVP support, continue with any remaining communication ops if needed.

Candidates to inspect:

- `all_gather`
- `reduce_scatter`
- `all_to_all`
- `ppermute`

Only add JVPs where semantics are clear and test them numerically.

---

## Strong recommendation for the next agent

Do **not** start by broadening op coverage again.

Start by improving the abstraction:

1. unify deferred-reduction representation,
2. improve transform-boundary tests,
3. then continue op/communication coverage.

That order is much more likely to produce a durable system.

---

## Minimal “get started” checklist for the next developer

1. Read:
   - [docs/dev/spmd_partial_architecture_review.md](docs/dev/spmd_partial_architecture_review.md)
   - this handover
2. Re-run:
   - the focused test commands above
3. Inspect:
   - [nabla/core/sharding/spmd.py](nabla/core/sharding/spmd.py)
   - [nabla/ops/communication/all_reduce.py](nabla/ops/communication/all_reduce.py)
   - [nabla/transforms/vmap.py](nabla/transforms/vmap.py)
   - [nabla/transforms/compile.py](nabla/transforms/compile.py)
4. Continue with:
   - unified effect model
   - new rigorous transform-composition tests

---

## Bottom line

The system is now in a meaningfully better state:

- less duplicated policy logic,
- better sharding metadata transport,
- stronger compile cache correctness,
- better forward-mode support,
- fewer ignored sharding failures.

But the deepest architectural issue remains:

> deferred reduction is still represented in two different internal forms.

That is the next real step if the goal is to make Nabla's sharding compiler future-proof, readable, and maintainable.
