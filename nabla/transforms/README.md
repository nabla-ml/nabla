# Function Transforms

[← Back to Root](../README.md)

## Philosophy
Transforms wrap a user function to alter its execution semantics. They are the bridge between "Logical Code" (what the user writes) and "Physical Execution" (what runs on hardware).

## Architecture & Internals

### `shard_map` (Distributed Execution)

Executes single-device code across a device mesh using SPMD parallelism.

**Three-Phase Process**:

1. **Trace Logical Function**: Execute function with logical tensor inputs to capture operation graph.

2. **Auto-Sharding (Optional)**: If `auto_sharding=True`:
   - Extract graph topology and operation costs
   - Run `SimpleSolver` to determine optimal sharding strategy
   - Apply solver constraints to the graph

3. **Trace-and-Replay**: Execute function again with physical tensors:
   - Each logical tensor becomes a collection of shard tensors
   - Operations automatically invoke sharding propagation
   - Communication ops inserted by `reshard_inputs` where needed
   - Result is fully distributed computation graph

**Key Insight**: User writes single-device math. Sharding propagation and communication insertion happen automatically during physical trace replay.

### `vmap` (Automatic Batching)

Vectorizes a function over a batch dimension without explicit loops.

**Prefix Semantics**: Batch dimensions are always leading dimensions in the tensor shape. For nested vmaps, multiple batch dimensions stack at the front.

**Batch Dimension Propagation**:
- Operations receive tensors with `batch_dims` metadata
- Binary ops unify batch dimensions: `output_batch_dims = max(lhs_batch_dims, rhs_batch_dims)`
- Broadcasting handled via `broadcast_batch_dims` internal operation
- Reductions over batch dimensions require special handling

**Implementation**:
- `TensorImpl` tracks `batch_dims` count
- View operations (`incr_batch_dims`, `decr_batch_dims`) manipulate batch dimension metadata
- Physical shapes include batch dimensions as leading axes

Example:
```python
@vmap
def f(x, y):
    return x * y + x

# Input shapes: x=[10, 5], y=[10, 5]
# Batch dimension: 0 (size 10)
# Inner computation sees: x=[5], y=[5]
# Output: [10, 5]
```

### `compile` (JIT Compilation)

Just-in-time compilation with graph optimization.

**Compilation Pipeline**:

1. **Graph Capture**: Trace function to build computation graph
2. **Optimization Passes**:
   - **Dead Code Elimination (DCE)**: Remove unused operations
   - **Common Subexpression Elimination (CSE)**: Deduplicate identical subgraphs
   - **Constant Folding**: Pre-compute constant expressions
3. **Code Generation**: Lower graph to MAX executable
4. **Caching**: Store compiled artifact, keyed by function identity and input shapes
5. **Execution**: Run compiled code on subsequent calls

**Benefits**:
- Eliminates Python interpreter overhead
- Enables whole-graph optimizations
- Amortizes compilation cost across multiple invocations

**Compatibility**: Works with both sharded and non-sharded tensors. When combined with `shard_map`, compiles the distributed graph with all communication operations inlined.

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
