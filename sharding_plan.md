The user has created the following implementation plan to make the shard_map transformation in nabla/transforms/shard_map.py a working automated sharding system. However, this is still in its early stages and needs to be further developed. The user is quite happy with what we have right now. This system relies on the fact that each operation (if the inputs are sharded) correctly uses its local sharding_rule to determine whether after a respective spmd-call on the respective maxpr-method, we need any resharding or communication. The shard_map trafo itself retrieves an (probably unsharded) trace from the original function, then passes it to the solver as a JSON with mesh and shape annotations, and sharding constraints, then the solver returns a sharding plan, which is then applied to the original trace by replaying each operation in the trace given the now specified sharding constraints on the input tensors and output tensors. However the user is unsure if this system actually works correctly and want the new Agent to rigorously investigate this behavior. The user wants the solver to work in a forward and backward way, and want it to be able to use all the information available by our sharding representation/propagation as stated in the nabla/sharding module. The user wants the Agent to read the respective CLAUDE.md files and the respective python files to gather the correct context, and to do the appropriate research.


Here is the original implementation plan after we added the shard_map trafo, and before we started working on the actually auto-sharding capabilities/solver/JSON-representation:


START:

# Automated Sharding System Design

This document outlines the design and architecture of the Automated Sharding System in `nabla`. This system transforms `shard_map` from a manual annotation tool into an intelligent optimizer that automatically determines the optimal sharding strategies for operations based on a device mesh.

## System Architecture

The system is composed of three main components:

1.  **Graph Extractor** (`nabla/transforms/graph_extractor.py`)
2.  **Solver** (`nabla/optimizer/simple_solver.py`)
3.  **Integration Layer** (`nabla/transforms/shard_map.py`)

### 1. Graph Extractor

The `ShardingGraphExtractor` class is responsible for converting the logical trace of specific `shard_map` functions into a framework-agnostic, unsharded graph representation (JSON).

**Key Responsibilities:**
-   **Trace Analysis**: It iterates over the captured trace.
-   **Metadata Extraction**: Extracts input/output shapes, dtypes, and tensor identifiers.
-   **Rule Instantiation**: For each operation, it instantiates the corresponding `sharding_rule` to determine factorization (e.g., `(m, k) -> (m)`) and valid parallelism axes.
-   **Constraint Capture**: Captures any user-defined sharding constraints present on inputs or outputs.
-   **Serialization**: outputs a JSON string representing the compute graph.

### 2. Simple Solver

The `SimpleSolver` implements the optimization logic. It takes the JSON graph and the device mesh as input and produces a sharding plan.

**Optimization Strategy (Greedy):**
-   The current implementation uses a **greedy, node-by-node optimization** approach.
-   For each node (operation), it evaluates candidate strategies (e.g., Data Parallelism vs. Model Parallelism for Matmul).
-   **Cost Model**: A pluggable cost model (in `nabla/ops`) calculates the theoretical cost of each strategy:
    -   `Compute Cost`: FLOPs / Parallelism Factor.
    -   `Communication Cost`: Bytes Transferred * Link Latency / Bandwidth.
-   **Selection**: The strategy with the minimum total cost is selected.

**Node-Centric Output:**
The solver produces a hierarchical solution structured as:
```json
{
  "nodes": {
    "node_id": {
      "inputs": {
        "input_index": { "dims": [...], "replicated": [...] }
      },
      "outputs": {
        "output_index": { "dims": [...], "replicated": [...] }
      }
    }
  }
}
```
**Rationale**: By defining constraints *per-node input* rather than just *per-tensor*, the system can handle conflicting requirements (e.g., a "Fork" scenario where one consumer needs a tensor sharded and another needs it replicated).

### 3. Integration Layer (Shard Map)

The `shard_map` function acts as the orchestrator. When `auto_sharding=True` is set:

1.  **Extract**: It calls `ShardingGraphExtractor` to get the JSON graph.
2.  **Solve**: It invokes `SimpleSolver` to get the sharding plan.
3.  **Replay & Apply**:
    -   It replays the captured trace.
    -   Before executing each operation, it checks the **Input Constraints** from the solver's plan for that specific node.
    -   If a constraint exists, it explicitly inserts a `shard` operation (reshard) to transform the input tensor into the required layout (e.g., from Replicated to Split-K).
    -   This automatic resharding ensures that each operation runs in its locally optimal configuration, even if it requires data movement.

## Debuggability features

To assist in development and performance tuning, the system includes comprehensive debug instrumentation:

-   **`debug=True` Flag**: Added to `shard_map` (and propagated to Extractor/Solver).
-   **Graph Visualization**: Prints the extracted JSON graph structure and shapes.
-   **Solver Trace**: Prints detailed cost analysis for each node (e.g., `Costs: DP=6.71e+07, MP=4.26e+09`) and the selected strategy.
-   **Solution Dump**: Prints the final node-centric solution JSON.

## Verification & Testing

To verify the system's correctness and robustness, we have implemented the following test suites:

### 1. Unit Tests (`tests/unit/test_auto_sharding.py`)
These tests verify individual components in isolation:
-   **Graph Extractor**: Converts a known trace (Matmul -> Add) into the expected JSON structure.
-   **Solver Logic**: Feeds mock JSON graphs to `SimpleSolver` to ensure it selects the minimum cost strategy (e.g., preferring Data Parallelism when costs favor it).
-   **End-to-End Integration**: Runs `shard_map(..., auto_sharding=True)` and verifies the output tensor attributes.

### 2. Manual Verification Scripts
Used during development to trace complex scenarios (now documented in `walkthrough.md`):
-   **Fork Scenario**: Verified that the solver correctly handles conflicting requirements (Input A shared by Node 1 [needs Split] and Node 2 [needs Replicated]) by inserting local `shard` (reshard) operations.
-   **Multi-Output Operations**: Verified `split` op support where a single node produces multiple outputs with different downstream constraints.

To run the verification suite:
```bash
python -m unittest tests/unit/test_auto_sharding.py
```

## Onboarding for New Agents

If you are a new agent or engineer looking to understand or extend this system, follow this reading order:

1.  **High-Level Context**: Read `nabla/transforms/CLAUDE.md` to understand `shard_map`'s "Trace-and-Replay" mechanism (Dual Execution). This is the foundation upon which Auto Sharding is built.
2.  **Sharding Fundamentals**: Read `nabla/sharding/shardy_propagation.md` (if available) or `nabla/sharding/cl_sharding.py` to understand the `ShardingSpec` and `DeviceMesh` abstractions.
3.  **This Document**: Review the Architecture section above.
4.  **Code Walkthrough**:
    -   `nabla/transforms/graph_extractor.py`: How we "lift" the trace to a graph.
    -   `nabla/optimizer/simple_solver.py`: The optimization logic.
    -   `nabla/transforms/shard_map.py`: The `wrapper` function where integration happens.

## Future Directions & Missing Pieces

While the current system is functional and verified for basic workflows, several areas need development to reach production quality:

1.  **Advanced Cost Model**:
    -   **Current**: Heuristic-based (FLOPs vs. idealized Comm bytes).
    -   **Missing**: Real device mesh characteristics (Bandwidth, Latency), precise FLOP counts for all ops.
2.  **Broader Op Support**:
    -   **Current**: Optimized for `matmul`. Other ops use default/fallback strategies.
    -   **Missing**: Specific rules/costs for `conv`, `reduce_sum`, `reshape`, and `transpose` (complex views).
3.  **Global Optimization**:
    -   **Current**: Greedy node-by-node.
    -   **Missing**: Dynamic Programming (DP) or ILP to optimize the entire graph globally, accounting for the "chain reaction" of resharding costs.
4.  **Framework Integration**:
    -   **Current**: Standalone Nabla.
    -   **Missing**: Adapters to ingest graphs from JAX/PyTorch or lowering to XLA/StableHLO sharding annotations.

END

End of the original implementation plan, which has already been implemented partially, but still needs a lot of work to be turned into a reliable auto-sharding system. We stick to a primitive, greedy, !bidirectional!, Python-based solver for now, but expect this to change to a proper, ILP-based, dynamic-programming-based, C++-based solver in the future. We should plan for this type of solver now already, and think about how to represent the problem accordingly!

If the Agent thinks that we need to change things about the implementation plan, it should try to convince the user so verbally first, then create its own new implementation plan as an artifact, and then ask the user to proceed with this plan.

General goals: The agent should create concise, readable and elegant code, which works nicely together with what the library provides already.