# Nabla (nabla_ml) Context

## Project Overview
Nabla is a Machine Learning library for the Python/Mojo ecosystem. It aims to provide:
-   **PyTorch-like API**: Imperative `Tensor` operations and `.backward()`.
-   **JAX-like Transforms**: Purely functional transformations like `grad`, `vmap`, `jit`.
-   **Unified SPMD**: "Lazy-Eager" execution where code written for a single device is automatically sharded and parallelized across devices (TPUs/GPUs).

**Current Status**: Foundation Phase. The project is undergoing a rewrite to establish a solid core for distributed execution (Sharding) and graph compilation. Focus is on depth and reliability over feature breadth.

## Architecture
The library is structured into three main components within the `nabla/` directory:

1.  **`core/` (The Engine)**:
    -   **`tensor/`**: State management (`TensorImpl`, `Tensor`).
    -   **`graph/`**: Operation recording and tracing (`ComputeGraph`).
    -   **`sharding/`**: Compiler pass for distributed execution.
    -   **`common/`**: Shared utilities (`pytree`, context).
    -   **Rule**: Strict dependency layering (`sharding` -> `tensor` -> `graph` -> `common`) to avoid circular imports.

2.  **`ops/` (The Logic)**:
    -   Defines math operations (`Add`, `Matmul`), their gradients (`vjp`), and sharding rules.

3.  **`transforms/` (The Bridge)**:
    -   Implements functional transforms: `vmap` (vectorization), `shard_map` (manual distribution), `compile` (JIT).

**Execution Flow**:
1.  **Interact**: User writes eager code using `Tensor`.
2.  **Trace**: Operations are recorded into a `ComputeGraph`.
3.  **Compile**: Data access triggers compilation.
4.  **Shard**: Graph is annotated with physical sharding specs (SPMD).
5.  **Execute**: Optimized graph runs on the backend.

## Development Workflow

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

### Testing
-   **Run all tests**:
    ```bash
    pytest
    ```
-   **Tests Directory**: `tests/` contains `unit`, `integration`, and `verification` tests.

### Code Quality
-   **Linting**:
    ```bash
    ruff check .
    ```
-   **Formatting**:
    ```bash
    black .
    ```
-   **Type Checking**:
    ```bash
    mypy
    ```

## Crucial Conventions
1.  **Documentation Maintenance**: Submodules (like `nabla/core`, `nabla/ops`) have their own `README.md` files. **You MUST update these READMEs whenever you modify the corresponding architecture or module structure.** They serve as the source of truth.
2.  **Lazy-Eager Pattern**: Remember that despite the eager API, the underlying execution is lazy and compiled. Changes often involve the graph tracing logic, not just immediate execution.
3.  **No Circular Imports**: Respect the `core` layering.
