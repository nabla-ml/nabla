# Nabla: Advanced Agentic Coding Framework

> **⚠️ CRITICAL**: Always activate the virtual environment first:
> `source venv/bin/activate`

## Overview
Nabla is a **Lazy-Eager** tensor execution framework for MAX, combining an imperative PyTorch-like API with a JAX-style distributed compilation engine.

## Documentation Map

| Module | Description | Guide |
| :--- | :--- | :--- |
| **`nabla/`** | **Core Architecture**. Lazy-eager logic, tensor internals. | [Architecture](nabla/CLAUDE.md) |
| **`nabla/core/`** | **Internals**. Tensor state, graph tracing, output refs. | [Internals](nabla/core/CLAUDE.md) |
| **`nabla/ops/`** | **Operations**. Singleton pattern, dispatch logic. | [Ops Guide](nabla/ops/CLAUDE.md) |
| **`nabla/transforms/`** | **Transforms**. Vmap, JIT compilation, Shard Map. | [Transforms](nabla/transforms/CLAUDE.md) |
| **`nabla/sharding/`** | **Distributed**. SPMD, Device Mesh, Propagation. | [Sharding](nabla/sharding/CLAUDE.md) |

## Quick Start
```bash
# Run all unit tests
python -m pytest tests/unit

# Run key sharding tests
python -m pytest tests/unit/test_auto_sharding.py -v
```

## Key Test Entry Points
-   **Sharding Logic**: [`tests/unit/test_auto_sharding.py`](file:///Users/tillife/Documents/CodingProjects/nabla/tests/unit/test_auto_sharding.py)
-   **Complex Chains**: [`tests/unit/test_sharding_stress.py`](file:///Users/tillife/Documents/CodingProjects/nabla/tests/unit/test_sharding_stress.py)
-   **Communication**: [`tests/unit/test_communication_ops.py`](file:///Users/tillife/Documents/CodingProjects/nabla/tests/unit/test_communication_ops.py)
