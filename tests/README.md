# Testing Strategy

[← Back to Root](../AGENTS.md)

## Philosophy
Nabla tests must verify the "Dual" nature of the framework:
1.  **Logical Correctness**: Does `C = A @ B` produce the right numbers?
2.  **Physical Correctness**: Does `shard_map` split A and B correctly across devices?
3.  **Propagation Correctness**: Does the Auto-Sharding solver find optimal layouts?

## Test Categories

### Unit Tests (`tests/unit/`)
*   `test_auto_sharding.py`: **Crucial**. Verifies the constraint solver, graph extraction, and full E2E execution of `shard_map(auto_sharding=True)`.
*   `test_sharding_spec.py`: Tests the sharding data structures (`DimSpec`, `ShardingSpec`).
*   `test_communication_ops.py`: Verifies low-level collective ops (`all_gather`, `all_reduce`) work on the mocked mesh.

### Integration Tests (`tests/integration/`)
*   `test_pp_transformer.py`: Complex pipeline parallelism scenarios.

## Key Test Files

| File | Role | Key Concepts |
| :--- | :--- | :--- |
| `conftest.py` | **Fixtures**. | Sets up `mock_mesh` for distributed tests. |
| `test_auto_sharding.py` | **The Brain**. | Tests the entire compiler stack (Trace -> Solver -> Replay). |
| `test_sharding_stress.py` | **Stability**. | Runs complex chains (diamond patterns, multi-op) to ensure fixed-point convergence. |

## Guidance for New Tests
When adding a new Operation:
1.  **Logical Test**: Ensure it computes correctly on CPU/Single Device.
2.  **Sharding Test**: Add a case in `test_auto_sharding.py` to verify it propagates factors correctly.
3.  **Stress Test**: If the op involves complex communication, consider adding a case to `test_sharding_stress.py`.

## Maintenance Guide
> **Note to AI Agents**:
> 1.  **Update Requirement**: You **MUST** update this file whenever you add new test categories, key test files, or change the testing strategy.
> 2.  **Accuracy**: This file serves as the source of truth for the testing architecture.
