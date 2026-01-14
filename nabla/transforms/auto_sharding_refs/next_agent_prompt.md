# Sharding System Improvement - Agent Prompt

Read the sharding-related documents and files in this project to understand the architecture, then rigorously verify and extend the system's robustness.

## Phase 1: Required Reading (Do This First!)

1. **CLAUDE.md Files** - Read these for context on each module:
   - `nabla/sharding/CLAUDE.md` - Factor-based propagation, Shardy philosophy
   - `nabla/transforms/CLAUDE.md` - shard_map trace-and-replay mechanism
   - `nabla/ops/CLAUDE.md` - Operation ABC hierarchy, sharding rules

2. **Research Documents**:
   - `nabla/transforms/auto_sharding_refs/sharding_plan.md` - Current status and roadmap
   - `nabla/transforms/auto_sharding_refs/alpa_paper_summary.md` - ALPA algorithm overview
   - `nabla/transforms/auto_sharding_refs/alpa_integration_research.md` - GSPMD vs Shardy comparison

3. **Core Implementation Files**:
   - `nabla/ops/operation.py` - ABC hierarchy with `compute_cost()`
   - `nabla/ops/communication.py` - CollectiveOperations with `communication_cost()`
   - `nabla/sharding/propagation.py` - Factor-based sharding propagation
   - `nabla/optimizer/simple_solver.py` - Cost-based DP vs MP strategy selection
   - `nabla/transforms/shard_map.py` - Entry point for auto-sharding

## Phase 2: Research and Verification

1. **Run the existing test suites** with debug output to understand current behavior:
   ```bash
   python -m pytest tests/unit/test_auto_sharding.py -v
   python -m pytest tests/unit/test_cost_model.py -v
   python -m pytest tests/unit/test_communication_ops.py -v
   ```

2. **Trace through an E2E example** manually to understand the flow:
   - How `shard_map` traces a function
   - How the solver selects DP vs MP
   - How factor propagation spreads constraints
   - Where AllReduce gets inserted for sharded contracting dimensions

3. **Identify gaps** in current implementation by comparing with ALPA's requirements

## Phase 3: Implementation Focus Areas

### Priority 1: Memory Estimation
ALPA's ILP needs memory estimates per tensor per sharding strategy. Currently missing.
- Add `memory_cost()` to Operation base class
- Implement for key ops (matmul produces M*N output, etc.)
- Integrate into solver for memory-aware strategy selection

### Priority 2: Sharding Rules Coverage
Many ops lack proper `sharding_rule()` implementations or tests:
- Audit all ops in `view.py` (reshape, transpose, slice, concat)
- Add/verify sharding rules for `attention`, `layernorm` if they exist
- Create tests that verify propagation through complex op sequences

### Priority 3: Numerical Verification
Add tests that compare sharded vs unsharded execution:
- Run same function with and without `shard_map`
- Verify numerical equivalence (np.allclose)
- Test edge cases: uneven shards, broadcasting, reductions

### Priority 4: Cross-Mesh Investigation
Document what would be needed for cross-mesh propagation (e.g., CPU data loading mesh → GPU training mesh).

## Success Criteria

- [ ] All existing tests still pass
- [ ] Memory estimation implemented and tested
- [ ] At least 3 new complex graph tests (multi-attention, residual connections)
- [ ] Documentation updated in CLAUDE.md files
- [ ] No double-counting of communication costs in any scenario
