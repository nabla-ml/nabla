# Nabla Architecture

> [!IMPORTANT] Project Status: Foundation Phase
> Nabla is fundamentally a **Training Framework**. However, we are currently in the **Foundation Phase**, focused exclusively on:
> 1.  **Forward (Sharded) Execution**: Ensuring the distributed compiler works perfectly.
> 2.  **Core Building Blocks**: Making the core engines and transforms (`vmap`, `shard_map`, `compile`, etc.) reliable.
>
> We are currently prioritizing **Depth and Reliability** of the core system over **Breadth**. Features like extensive operator support, high-level NN modules, and syntactic sugar are explicitly deferred.

## Philosophy

Nabla implements **"Lazy-Eager" Execution** on a **Unified SPMD Runtime** with **Factor-Based Sharding**.

**Lazy-Eager Execution**: Code looks eager (like PyTorch), but operations are lazily traced into a computation graph and compiled just-in-time when you access data.

**Unified SPMD Runtime**: You write code for a single logical device. The compiler automatically parallelizes it across a cluster of accelerators through sharding propagation.

**Factor-Based Sharding**: Unlike JAX's dimension-to-axis mapping, Nabla maps **named factors** (abstract indices in operations) to device mesh axes. This handles broadcasting, reshaping, and einsum-style operations naturally without dimension tracking complexities.

### Execution Model: Logical vs Physical

Operations in Nabla execute in two layers:

**Logical Layer** (User Interface):
- Validates inputs, handles broadcasting and type promotion
- Performs pre-operation data movement via `preshard_inputs`
- Delegates computation to physical layer
- Wraps results into `nabla.Tensor` objects
- Enables tracing with symbolic nodes

**Physical Layer** (SPMD Kernel Runner):
- Executes actual computation on device shards
- Signature: `physical_execute(op, args, kwargs, mesh) -> PhysicalResult`
- Loops over each shard and calls `op.maxpr()` (MAX Primitive)
- Must run inside `graph.context()` for lazy value access
- Handles per-shard argument transformations

This separation ensures robust trace rehydration: traced graphs can be replayed without triggering recursive propagation, and physical execution remains independent of the tracing system.

## Architecture & Internals

### The Lifecycle

1. **Interact**: User manipulates `Tensor` objects (pointers to graph nodes).
2. **Trace**: Operations like `x + y` record nodes in the global `ComputeGraph`.
3. **Compile**: Accessing data (e.g., `print(x)`) triggers compilation.
4. **Shard**: The compiler runs **Sharding Propagation** to determine per-shard execution.
5. **Execute**: The graph executes on accelerators via MAX Engine.

### Sharding Propagation: The Three-Phase Algorithm

For each operation, sharding propagates through three phases:

**COLLECT**: Convert dimension shardings to factor shardings. For `matmul(A, B)` with sharding rule `"m k, k n -> m n"`, factor `k` collects sharding from both input dimensions.

**RESOLVE**: Resolve conflicts using priority system. Explicit replications override higher priorities. Higher parallelism wins at equal priority. Detects contracting factors (present in inputs but not outputs) which become partial sum axes.

**UPDATE**: Project factor shardings back to output dimension shardings. Marks axes as `partial=True` if they hold unreduced partial sums from contracting factors.

### Communication Injection

After propagation, `reshard_inputs` automatically inserts communication operations when input shardings don't match required shardings:

- **AllReduce**: Reduce partial sums across axes (e.g., after matmul with sharded contracting dimension)
- **AllGather**: Gather sharded data to replicated
- **AllToAll**: Redistribute data across different mesh axes
- **ReduceScatter**: Reduce then scatter results

Example: Row-parallel matmul with `A` sharded on `k` and `B` sharded on `k` produces output with `partial_sum_axes={'k'}`, triggering automatic `AllReduce` on the `k` axis.

### The Dual System

Every sharded logical tensor is backed by a graph of physical shards. This allows reusing the same graph engine for both single-device and distributed execution, at the cost of complexity in the `shard_map` trace-and-replay engine.

## Module Map

| Module | Purpose | Documentation |
| :--- | :--- | :--- |
| **`core/`** | **The Engine**. Contains the `Tensor` state, `ComputeGraph` recorder, and the `Sharding` compiler. | [**Read Docs**](core/README.md) |
| **`ops/`** | **The Logic**. Defines all mathematical operations (`Add`, `Matmul`), their gradients, and sharding rules. | [**Read Docs**](ops/README.md) |
| **`transforms/`** | **The Bridge**. Functional transformations that alter execution: `vmap` (vectorize), `shard_map` (distribute), `compile` (optimize). | [**Read Docs**](transforms/README.md) |

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Read Recursively**: Start here, then follow links to understand specific subsystems.
> 2.  **Update Requirement**: You **MUST** update this file whenever you modify, restructure, or add ANY code in this module. Do not skip this step.
> 3.  **Accuracy**: This file serves as the source of truth for the module's architecture. Ensure the Component Map and Philosophy sections remain accurate after your changes.
> 4.  **Template**: Always use the `Philosophy` -> `Internals` -> `Map` -> `Maintenance` structure.
