# Automated Sharding System: Task Guide for Agents

> **Your Mission**: This system is functional but **needs rigorous verification and improvement**.
> Do NOT assume it works correctly. Your job is to test, verify, and strengthen it.

---

## ⚠️ Current Status: Needs More Testing

The auto-sharding system passes basic tests, but:

- **Only tested on simple cases** (single matmul, small chains)
- **Solver conclusions have not been rigorously verified** against expected behavior
- **Large multi-op graphs have not been tested**
- **Debug traces should be manually inspected** to confirm correctness

**Your first task**: Run with `debug=True` and **manually verify** the solver is making sensible decisions.

---

## Your Tasks (In Order)

### Task 1: Verify Solver Conclusions Are Correct

Run the existing tests with `debug=True` and study the output:

```bash
source venv/bin/activate
python -c "
from tests.unit.test_auto_sharding import TestAutoSharding
t = TestAutoSharding()
t.setUp()
t.test_integration_e2e()
"
```

**Questions to answer**:
- Does the solver choose DP (split batch) or MP (split model) correctly?
- Are the propagation iterations making logical sense?
- Does the fixed-point converge to a reasonable solution?
- Are the debug traces showing the sharding you expect?

### Task 2: Test on Larger Graphs

The current tests only use 1-3 operations. Create a test with a **real model architecture**:

```python
def transformer_block(x, w_q, w_k, w_v, w_o, w_ff1, w_ff2):
    # Attention
    q = x @ w_q
    k = x @ w_k  
    v = x @ w_v
    # (simplified - real attention would have softmax etc.)
    attn = q @ k.T @ v
    attn_out = attn @ w_o
    
    # FFN
    h = (x + attn_out) @ w_ff1
    h = h @ w_ff2
    
    return x + h
```

**Verify**:
- Does sharding propagate correctly through 8+ operations?
- Are there any nodes without solutions?
- Does the solver converge within a reasonable number of iterations?
- Is numerical output correct vs NumPy reference?

### Task 3: Verify Communication Ops Are Inserted Correctly

Check that `all_reduce` is inserted when sharding contracting dimensions:

```python
# Model Parallel matmul should trigger all_reduce
def mp_matmul(a, b):
    return a @ b  # If k-dim is sharded, need all_reduce

# Trace this and look for:
# - Solver choosing to shard K dimension
# - all_reduce appearing in the trace
```

### Task 4: Audit Each Operation Type

Create tests for each operation type to verify sharding rules work:

| Op | Test Needed |
|----|-------------|
| `matmul` | ✅ Tested |
| `add` | ✅ Tested (in chain) |
| `reduce_sum` | ⚠️ Needs standalone test |
| `transpose` | ❌ Not tested |
| `reshape` | ❌ Not tested |
| `softmax` | ❌ Not tested |
| `layer_norm` | ❌ Not tested |

### Task 5: Stress Test the Fixed-Point Propagation

Create adversarial cases:

```python
# Diamond pattern - does propagation handle forks correctly?
def diamond(x, w1, w2):
    h = x @ w1
    branch1 = h + 1
    branch2 = h * 2
    return branch1 @ branch2.T
```

### Task 6: Compare Against Unsharded Execution

Create a test that runs the SAME function:
1. Without auto_sharding (manual in_specs)
2. With auto_sharding

**Verify**: Results match AND solver chose a sensible strategy.

---

## How to Navigate the Codebase

### Essential Reading Order

1. **`nabla/sharding/CLAUDE.md`** - Sharding concepts (DeviceMesh, ShardingSpec, factors)
2. **`nabla/transforms/CLAUDE.md`** - shard_map trace-and-replay mechanism
3. **`nabla/optimizer/simple_solver.py`** - The solver you're verifying
4. **`nabla/sharding/propagation.py`** - The propagation algorithm

### Key Files

| File | What It Does | What to Check |
|------|--------------|---------------|
| `simple_solver.py` | Finds sharding strategy | Is the 4-phase algorithm correct? |
| `shard_map.py` | Orchestrates execution | Are constraints applied correctly? |
| `propagation.py` | Spreads sharding through ops | Does bidirectional flow work? |
| `graph_extractor.py` | Creates JSON graph | Are all ops extracted correctly? |
| `communication.py` | All collective ops | Are ops called when needed? |

---

## Debugging: What to Look For

### Good Debug Output (Solver working)

```
[Solver] Analyzing Node 0: matmul
  > Costs: DP=6.71e+07, MP=4.26e+09
  > Selected Strategy: Data Parallel (Split M)
[Solver] Propagation iteration 1: changed=True
[Solver] Propagation iteration 2: changed=False
[Solver] Fixed-point reached after 2 iterations
```

### Bad Signs (Investigate!)

```
[Solver] WARNING: Did not converge after 100 iterations
[shard_map] WARNING: No solution for node 3
[GraphExtractor] WARNING: Failed to get sharding_rule for node X
```

---

## Known Limitations (Be Skeptical!)

| Area | Concern | What to Verify |
|------|---------|----------------|
| Cost model | Only FLOPs/bytes heuristic | Does it make sensible choices? |
| Op coverage | Only matmul has cost heuristics | Do other ops propagate correctly? |
| Large graphs | Untested | Does fixed-point converge? |
| Complex patterns | Untested | Forks, diamonds, branches? |
| Numerical accuracy | Basic checks only | Test with float16? Large tensors? |

---

## How the System Works (Brief)

```
1. trace(func) → Captures logical graph
2. GraphExtractor → Converts to JSON with sharding rules
3. SimpleSolver:
   - Phase 1: Parse JSON → ShardingSpecs
   - Phase 2: Seed strategic ops (matmul DP/MP choice)
   - Phase 3: Fixed-point propagation until stable
   - Phase 4: Export node-centric solution
4. Replay → Execute with solver constraints, insert communication
```

### The Key Insight

Each op has a `sharding_rule()` returning einsum-like notation:
```
matmul: "m k, k n -> m n"
```

Factor `k` is "contracting" (input-only) → if sharded, needs AllReduce.

This is detected automatically. BUT: **Verify this actually happens!**

---

## Tests Currently Passing

```bash
python -m pytest tests/unit/test_auto_sharding.py -v
# 7 passed
```

But these are **basic tests**. A rigorous agent should:
1. Add larger graph tests
2. Add per-op-type tests
3. Manually inspect debug output
4. Verify all edge cases

---

## Success Criteria

You can be confident the system works when:

1. [ ] Large transformer-like graph produces correct numerical output
2. [ ] Debug traces show sensible sharding decisions for each op
3. [ ] All operation types have passing tests
4. [ ] Fork/diamond patterns work correctly
5. [ ] Solver always converges (no 100-iteration warnings)
6. [ ] Communication ops appear exactly when expected

---

## Running Tests

```bash
# Always use venv!
source venv/bin/activate

# Run auto-sharding tests
python -m pytest tests/unit/test_auto_sharding.py -v

# Run single test with debug output
python -c "
from tests.unit.test_auto_sharding import TestAutoSharding
t = TestAutoSharding()
t.setUp()
t.test_multi_op_chain()
"
```

---

**Remember**: This system is promising but UNPROVEN at scale. Test rigorously. Question everything. Verify with debug traces.