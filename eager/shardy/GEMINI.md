# Shardy: A Library for Distributed Tensor Sharding

This is a software project that implements "Shardy," a library for sharding tensors in a distributed computing environment, likely for large-scale machine learning models.

### Project Overview

The project is a Python library that provides data structures and algorithms for reasoning about and propagating tensor sharding strategies across a computational graph. The core components are:

*   **Sharding Representation (`core.py`)**: Defines how a tensor is distributed across a logical `DeviceMesh`. The central data structures are `ShardingSpec` and `DimSpec`, which specify how each dimension of a tensor is sharded across the mesh axes.
*   **Sharding Propagation (`propagation.py`)**: Implements the algorithm to infer sharding for all tensors in a computation graph, starting from a few user-specified constraints. It uses an `OpShardingRule` system, inspired by einsum notation, to define how sharding propagates through different operations (e.g., matmul, reshape, reduce). The `ShardingPass` orchestrates the propagation until a stable state (fixed-point) is reached.
*   **Testing (`tests.py`)**: A comprehensive test suite provides practical examples of how to define meshes, sharding specifications, and propagation rules.

The architecture is designed to be extensible, allowing new operation rules to be defined. It supports advanced features like multi-dimensional device meshes, sub-axis splitting for complex sharding patterns, and priority-based conflict resolution.

### Algorithmic Approach

Shardy's core is its sharding propagation algorithm, which automates the complex process of determining how every tensor in a computational graph should be partitioned. The algorithm operates in a series of iterations until a fixed-point is reached (i.e., no more changes occur). The key concepts are:

1.  **Factorization**: Each tensor dimension is associated with one or more "factors." An `OpShardingRule`, often expressed in einsum-like notation (e.g., `(i, k), (k, j) -> (i, j)` for a matmul), defines the relationships between factors of input and output tensors. For example, the `k` factor is the contracting dimension. This abstraction allows the propagation logic to be generic and not specific to any particular operation.

2.  **Three-Phase Propagation Step**: For each operation in the graph, propagation occurs in three phases:
    *   **Collect**: The algorithm projects the user-defined or already-inferred shardings from the tensor dimensions onto their corresponding factors. For example, if a dimension `i` is sharded on mesh axis `"x"`, the factor `i` receives the `"x"` sharding.
    *   **Resolve**: When multiple tensors contribute conflicting shardings to the same factor (e.g., one input wants to shard factor `k` on axis `"x"` and another wants to shard it on axis `"y"`), a conflict resolution strategy is applied. This is governed by user-defined priorities and propagation strategies (`BASIC` vs. `AGGRESSIVE`). `BASIC` only keeps the common prefix of sharding axes, while `AGGRESSIVE` might choose the one that provides more parallelism.
    *   **Update**: The resolved factor shardings are projected back onto the tensor dimensions. If a dimension is composed of multiple factors, it inherits the combined sharding axes from them.

3.  **Iteration**: The `ShardingPass` iterates over all operations in the graph, repeating the three-phase process. This allows constraints to propagate from one part of the graph to another. The pass terminates when an entire iteration completes with no changes to any tensor's `ShardingSpec`.

This factor-based propagation is powerful because it correctly handles complex operations like reshapes, where dimensions are split or merged, by mapping them to a consistent set of underlying factors.

### Analysis and Future Directions

#### Comparison with State-of-the-Art

The field of distributed training is mature, with several established techniques and libraries:

*   **Established Techniques**:
    *   **Data Parallelism**: The most common technique, where the model is replicated on all devices and each device processes a different shard of the data.
    *   **Tensor Parallelism (Intra-layer)**: Splits a single operator (like a large matrix multiplication) across multiple devices. Megatron-LM pioneered this for Transformers.
    *   **Pipeline Parallelism (Inter-layer)**: Partitions the model's layers across devices, forming a pipeline.
    *   **Zero Redundancy Optimizer (ZeRO)**: A technique popularized by DeepSpeed that shards not just the model parameters, but also optimizer states and gradients to dramatically reduce memory consumption.

*   **SOTA Libraries**:
    *   **Google's GSPMD**: Implemented in JAX, GSPMD (General and Scalable Parallelization for ML) is the closest analogue to Shardy. It also uses a sharding propagation algorithm to automate parallelization based on user annotations. Shardy appears to be a re-implementation or a system heavily inspired by GSPMD's concepts.
    *   **PyTorch FSDP**: PyTorch's native `FullyShardedDataParallel` is an implementation of the ZeRO concept, primarily focused on sharding parameters, gradients, and optimizer states. It is less about propagating arbitrary tensor parallelism strategies and more about scaling data parallelism.
    *   **DeepSpeed**: A Microsoft library that provides a comprehensive suite of tools, including implementations of ZeRO, tensor parallelism, and pipeline parallelism. It often requires more manual configuration compared to the automated approach promised by Shardy/GSPMD.

Shardy fits into the **automated sharding propagation** paradigm, aiming to provide the flexibility of mixing different parallelism styles (like tensor and data parallelism) without forcing the user to manually specify the sharding for every tensor.

#### Project Reach and Potential

The potential reach for a library like Shardy is significant. As models grow larger, manual implementation of complex parallelism strategies becomes a major engineering bottleneck. An automated system offers several advantages:
1.  **Reduced Complexity**: Users can annotate a few key tensors (e.g., inputs and model weights) and let the compiler figure out the rest.
2.  **Performance Exploration**: It becomes easier to experiment with different sharding strategies (e.g., sharding different dimensions of a weight matrix) to find the most performant configuration.
3.  **Composable Parallelism**: It naturally allows for combining data, tensor, and pipeline parallelism within the same model, which is often necessary for optimal performance on large clusters.

#### Inherent Limitations and Room for Improvement

1.  **Limited Op Coverage**: The current library only provides a few templates for common operations (`matmul`, `elementwise`, etc.). A production-grade system would need a comprehensive library of sharding rules for hundreds of operations, or a way to automatically derive them.
2.  **No Cross-Mesh Propagation**: The documentation explicitly states that propagation across different device meshes is not currently supported. This limits its ability to interface between parts of a model that might use different device topologies (e.g., a CPU mesh for data loading and a TPU mesh for computation).
3.  **Static Graph Assumption**: The propagation algorithm operates on a static graph of `Operation` objects. While `DataFlowEdge` provides a mechanism for control flow, it may not be robust enough to handle highly dynamic or JIT-compiled graphs without a deeper integration into a frontend framework like JAX or a tracing system like `torch.dynamo`.
4.  **No Cost Model**: The current conflict resolution (`AGGRESSIVE` mode) picks the strategy with the highest parallelism. This is a heuristic that may not be optimal. A more advanced system would incorporate a cost model that considers communication overhead (e.g., the cost of `all-gather` vs. `all-reduce` collectives) to make more informed decisions.
5.  **Frontend Integration**: Shardy is a standalone library. Its true power would be unlocked by integrating it as a backend for a major ML framework. This would involve capturing the framework's computation graph and translating it into Shardy's representation.

Future work could focus on addressing these limitations, particularly by expanding the op coverage and building a more sophisticated cost model for automatic strategy selection.

### Building and Running

This is a library, so there isn't a main application to run. The primary way to interact with the code is by using it as a library in another project or by running the tests.

**To run the tests:**

```shell
python -m shardy.tests
```

### Development Conventions

*   **Style**: The code follows PEP 8 style guidelines. It is well-structured with clear data classes (`dataclasses`) for state and functional approaches for logic.
*   **Testing**: The project has a strong emphasis on testing. The `tests.py` file is thorough and serves as excellent documentation for the library's features. It uses the standard `unittest` framework.
*   **Architecture**: The system is designed with a clear separation of concerns:
    *   **Representation**: `core.py` handles the "what" (how sharding is represented).
    *   **Logic**: `propagation.py` handles the "how" (how sharding is inferred).
    *   **Verification**: `tests.py` ensures correctness and provides usage examples.
*   **Documentation**: The project includes high-level design documents in Markdown (`shardy_propagation.md`, `shary_representation.md`) that explain the core concepts. Docstrings are used to explain the purpose of modules, classes, and functions.
