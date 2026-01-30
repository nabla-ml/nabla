# Function Transforms

[← Back to Root](../README.md)

## Philosophy
Transforms wrap a user function to alter its execution semantics. They are the bridge between "Logical Code" (what the user writes) and "Physical Execution" (what runs on hardware).

## Architecture & Internals

### `shard_map` (Distributed Execution)

Distributes single-device code across a device mesh using **trace-and-replay**:

**Execution Flow**:

1. **Logical Trace**: Execute function with logical tensor inputs
   - User calls like `x.shard(mesh, spec)` execute eagerly, creating sharded tensors
   - Operations execute and record in computation graph
   - Captures the sequence of operations with their shardings

2. **Auto-Sharding (Optional)**: If `auto_sharding=True`
   - Extract graph topology and operation costs to JSON
   - Run `SimpleSolver` to compute optimal sharding strategy
   - Solver produces constraints (target shardings per operation)

3. **Physical Trace Replay**: Re-execute function with dual tensors
   - Each tensor now has `.dual` attribute pointing to physical shards
   - Operations detect dual mode, use `tensor.dual` for per-shard execution
   - Solver constraints enforced via additional `.shard()` calls during replay
   - Same Python code executes, different execution path (physical shards instead of logical)
   - Result: Distributed computation graph with per-shard operations

**Key Insight - Trace Rehydration**: 
- First trace captures "what operations to do" (logical structure)
- Second trace executes "how to do them distributed" (physical execution)
- Both traces execute eagerly, but operate on different tensor representations
- This separation enables robust replaying without re-invoking Python logic

### `vmap` (Vectorization)

Auto-batches operations over leading dimension(s):

**Prefix Semantics**: Batch dimensions always appear as leading dimensions in shape. For nested `vmap`, multiple batch dimensions stack at front.

**Mechanism**:
- `TensorImpl.batch_dims` tracks count of leading batch dimensions
- Binary ops unify batch dims: `output_batch_dims = max(lhs_batch_dims, rhs_batch_dims)`
- Internal ops (`incr_batch_dims`, `move_axis_to_batch_dims`) manage batch dimension metadata
- Operations execute normally but interpret leading N dimensions as batch

**Example**:
```python
@vmap  # Vectorize over axis 0
def f(x, y):
    return x * y  # Inner computation

f(x=[10, 5], y=[10, 5])  # Batch size 10
# Inner function sees virtual shapes [5], [5]
# Output: [10, 5]
```

### `compile` (JIT Compilation)

Defers graph compilation until first execution, caches for subsequent calls:

**Pipeline**:
1. **Trace**: Capture function execution as computation graph
2. **Optimize**: Dead code elimination (DCE), common subexpression elimination (CSE), constant folding
3. **Lower**: Generate MAX executable from optimized graph
4. **Cache**: Store compiled artifact keyed by (function, input_shapes, input_dtypes)
5. **Execute**: Run compiled code, bypass Python on subsequent calls

**Compatibility**: Works with both sharded and non-sharded tensors. Combined with `shard_map`, compiles the distributed graph including all communication operations.

## Component Map

| File | Role | Exported Symbols |
| :--- | :--- | :--- |
| [`shard_map.py`](shard_map.py) | **Distribution** | **Functions**: `shard_map` |
| [`vmap.py`](vmap.py) | **Vectorization** | **Functions**: `vmap`<br>**Classes**: `AxisSpec` (internal helper often used in docstrings) |
| [`compile.py`](compile.py) | **JIT & Opt** | **Functions**: `compile`<br>**Classes**: `CompiledFunction`, `CompilationStats` |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 2.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
