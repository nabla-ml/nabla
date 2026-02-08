# Nabla Unified Op Signatures — Refactoring Plan

**Goal**: Refactor all `Operation` subclass *internal methods* (`kernel`, `vjp_rule`, `jvp_rule`, `sharding_rule`, `execute`, etc.) to share a single unified signature, enabling future Mojo porting where all ops implement a fixed trait interface with no dynamic signatures.

**Key Distinction**: The user-facing `__call__` keeps its ergonomic per-op signature (e.g., `matmul(x, y)`, `reshape(x, shape=...)`). Only internal dispatch methods are unified.

**Validated by**: JAX's `Primitive.bind(*args, **params)` pattern — the industry standard for tracing-based frameworks.

---

## Unified Internal Signature

### Type Aliases (defined in `nabla/ops/base.py`)

```python
# Tensor argument list (always flat, no nesting)
OpArgs = list["TensorValue"]      # for kernel level
OpArgsTensor = list["Tensor"]     # for vjp/jvp rule level

# Static metadata dictionary (only primitive types)
OpKwargs = dict[str, int | float | bool | str | list[int] | list[float]]

# Always returns a list, even for single-output ops
OpResult = list["TensorValue"]    # for kernel level
OpResultTensor = list["Tensor"]   # for vjp/jvp rule level
```

### Target Signatures

| Method | Current (varies per op) | Unified |
|--------|------------------------|---------|
| `kernel` | `(self, x, **kw)` / `(self, *args, **kw)` | `(self, args: list[TensorValue], kwargs: OpKwargs) -> list[TensorValue]` |
| `vjp_rule` | `(self, primals, cotangent, output)` | `(self, primals: list[Tensor], cotangents: list[Tensor], outputs: list[Tensor], kwargs: OpKwargs) -> list[Tensor \| None]` |
| `jvp_rule` | `(self, primals, tangents, output)` | `(self, primals: list[Tensor], tangents: list[Tensor], outputs: list[Tensor], kwargs: OpKwargs) -> list[Tensor \| None]` |
| `sharding_rule` | `(self, in_shapes, out_shapes, **kw)` | `(self, in_shapes: list[tuple[int,...]], out_shapes: list[tuple[int,...]], kwargs: OpKwargs)` |
| `compute_cost` | `(self, in_shapes, out_shapes)` | No change needed (already uniform) |

### `multiple_results` Flag

Each op declares `multiple_results: bool = False`. When `True`, the `list[TensorValue]` from `kernel` maps to individual output tensors. When `False`, the list must have exactly 1 element.

---

## Op Compatibility Assessment

| Op Category | # Ops | args | kwargs | Notes |
|---|---|---|---|---|
| Unary | ~25 | `[x]` | `{}` or `{axis: int}` | Softmax has axis |
| Binary | ~7 | `[x, y]` | `{}` | Broadcasting in `__call__` |
| Comparison | ~9 | `[x, y]` | `{}` | |
| Reduce | ~10 | `[x]` | `{axis: int, keepdims: int}` | bool→int(0/1) |
| View/Shape | ~15 | `[x]` | `{axis: int}` / `{shape: list[int]}` | |
| Matmul | 1 | `[x, y]` | `{}` | 1D promotion in `__call__` |
| Gather/Scatter | 2 | `[x, idx]`/`[x, idx, upd]` | `{axis: int, batch_dims: int}` | |
| Concatenate | 1 | `[t0, t1, ..., tN]` | `{axis: int}` | Flatten list into args |
| Split/Chunk/Unbind | 3 | `[x]` | `{num_splits: int, axis: int}` | `multiple_results=True` |
| Pad | 1 | `[x]` | `{pad_flat: list[int], mode: str, value: float}` | Flatten paddings |
| Slice | 2 | `[x]`/`[x, update]` | `{start: list[int], size: list[int]}` | Drop slice objects |
| Creation | ~9 | `[]` | `{shape: list[int], dtype: str, ...}` | No tensor inputs |
| Where | 1 | `[cond, x, y]` | `{}` | |
| Collective | ~10 | `[x]` | `{...}` | |
| Control Flow | 3 | N/A | N/A | Separate `ControlFlowOperation` |

---

## Implementation Phases

### Phase 1: Core Infrastructure + Kernel Unification

**1.1** Add type aliases and `multiple_results` flag to `Operation` base in `nabla/ops/base.py`.

**1.2** Refactor `Operation.__call__` pipeline: after user-facing `__call__` normalizes to `(args, kwargs)`, the internal pipeline works uniformly.

**1.3** Refactor all `kernel` methods to `kernel(self, args, kwargs) -> list[TensorValue]`.

**1.4** Refactor `_setup_output_refs` to store flat `list[TensorImpl]` + `OpKwargs`.

**1.5** Refactor `execute` to pass `(args, kwargs)` uniformly.

**Testing after Phase 1:**
- Run `python -m pytest examples/ -x -v` (all examples are integration tests)
- Run `python -m pytest tests/unit/test_unified.py -x -v` (unified op tests)

### Phase 2: VJP/JVP Rule Unification

**2.1** Refactor all `vjp_rule` signatures: `(self, primals: list, cotangents: list, outputs: list, kwargs: dict) -> list`.

**2.2** Refactor all `jvp_rule` signatures: same pattern.

**2.3** Update `BackwardEngine._process_node` to pass kwargs explicitly, remove `_unwrap_single`.

**2.4** Update `apply_jvp` in `nabla/ops/utils.py` to use unified signatures.

**Testing after Phase 2:**
- Run `python -m pytest tests/unit/test_vjp.py -x -v`
- Run `python -m pytest tests/unit/test_jvp.py -x -v`
- Run `python -m pytest examples/ -x -v`

### Phase 3: PyTree Boundary Isolation + Transform Updates

**3.1** Ensure PyTree flatten/unflatten only in `__call__` and transform entry points.

**3.2** Add `ControlFlowOperation` base class (separate from unified trait).

**3.3** Update sharding_rule signatures to receive `kwargs: OpKwargs` instead of `**kwargs`.

**Testing after Phase 3:**
- Run `python -m pytest tests/unit/test_transforms.py -x -v`
- Run `python -m pytest tests/unit/test_transforms_composition.py -x -v`
- Run `python -m pytest examples/ -x -v`

---

## Testing Strategy

**Primary validation**: `python -m pytest examples/ -x -v` — the examples exercise MLP training, pipeline parallelism, JAX comparison, and compiled execution. They cover forward, backward, vmap, and compile transforms end-to-end.

**Targeted unit tests**: Only run the specific test file related to the latest change:
- After kernel changes: `tests/unit/test_unified.py`
- After vjp changes: `tests/unit/test_vjp.py`
- After jvp changes: `tests/unit/test_jvp.py`
- After transform changes: `tests/unit/test_transforms.py`
- After multi-output: `tests/unit/test_multi_output.py`
- After reduction: `tests/unit/test_reduction.py`

**Never run the full test suite** — compilation is slow in Nabla/MAX.

---

## Key Design Decisions

1. **`_derivative` pattern**: Keep it, formalize as optional method returning `Optional[Tensor]` (not `NotImplemented` sentinel). Base class uses it in default `vjp_rule`/`jvp_rule`.

2. **Attribute mutation (tangent, dual, batch_dims)**: Defer to Mojo port. Keep as-is in Python.

3. **Lazy imports**: Skip — Python-only concern.

4. **DType/Device encoding**: Only stringify in `OpKwargs`. Keep proper types elsewhere.

5. **Control flow ops**: Separate `ControlFlowOperation` hierarchy — callables can't be primitive kwargs.

6. **Output type**: Always `list[TensorValue]`/`list[Tensor]` internally. `__call__` wraps back to user-expected structure.

7. **kwargs passed explicitly**: `vjp_rule`/`jvp_rule` receive `kwargs` as parameter instead of reading `output.op_kwargs`.

8. **STATELESSNESS**: ALL internal methods and all ops MUST be stateless and functionally pure. We use classes for ops merely as a way to inherit logic and as a namespace, NEVER as storage. Concretely:
   - No `self._cached_batch_dims` or any other mutable state on Operation instances.
   - All context needed by a method is passed as parameters (batch_dims threaded through, not cached on self).
   - Op instances are singletons used purely for dispatch — they must be safe to call concurrently without any shared mutable state.
