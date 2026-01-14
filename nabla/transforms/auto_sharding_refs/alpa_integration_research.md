# ALPA Integration Research: Deep Dive

> **Bottom Line**: Nabla follows **Shardy** (not GSPMD), which is more modern. ALPA was designed for GSPMD's dimension-based propagation. Nabla's factor-based approach is fundamentally different but **more powerful** - it handles reshape/split/concat correctly. An ALPA-like solver for Nabla would need to understand factors, not just dimensions.

---

## 1. GSPMD vs Shardy: Key Difference

| Aspect | GSPMD (ALPA's target) | Shardy (Nabla's approach) |
|--------|----------------------|---------------------------|
| **Propagation granularity** | Dimension-based | **Factor-based** (sub-dimensions) |
| **Reshape handling** | Heuristics, often needs communication | Factors track data flow through reshapes |
| **User control** | Implicit via cost model | Explicit open/closed dims + priorities |
| **Axis splitting** | Not native | Built-in sub-axis support `"x":(2)4` |

**Implication**: An ALPA-like ILP for Nabla must enumerate **factor shardings**, not dimension shardings. This is actually cleaner but requires different problem formulation.

---

## 2. Existing Cost Model Assessment

**Current implementation** (from `operation.py` and ops):

```python
# Base Operation - default returns 0 (negligible cost)
def cost_model(self, input_shapes, output_shapes) -> float:
    return 0.0

# MatmulOp (in binary.py)
def cost_model(self, input_shapes, output_shapes) -> float:
    m, k = input_shapes[0][-2], input_shapes[0][-1]
    n = input_shapes[1][-1]
    return 2 * m * k * n  # FLOPs
```

**What it provides**: Per-op FLOPs (compute cost).

**What's MISSING for ALPA ILP**:
| Need | Status | Notes |
|------|--------|-------|
| Compute cost `d_vi` | ✅ Have | `cost_model()` returns FLOPs |
| Communication cost `c_vi` | ❌ Missing | AllReduce/AllGather bytes × bandwidth |
| Resharding cost `R_vuij` | ❌ Missing | Cost to go from spec_i to spec_j |
| Memory estimation | ❌ Missing | Needed for pipeline memory constraints |

**Conclusion**: Cost model provides compute, but **lacks communication cost estimation** needed for ILP objective function.

---

## 3. What Solver Needs to Know (C++ Replication)

If solver is in C++, it needs these concepts replicated:

### 3.1 Must Replicate (Core Data Structures)

```cpp
// --- From spec.py ---
struct DimSpec {
    vector<string> axes;      // Mesh axes sharding this dimension
    bool is_open;             // Can be further sharded?
    int priority;             // Propagation priority (0 = highest)
};

struct ShardingSpec {
    string mesh_ref;          // Mesh name reference
    vector<DimSpec> dims;     // One per tensor dimension
    set<string> replicated;   // Explicitly replicated axes
};

// --- From propagation.py ---
struct OpShardingRule {
    // Factor mappings: dim -> [factor_names]
    vector<map<int, vector<string>>> input_mappings;
    vector<map<int, vector<string>>> output_mappings;
    map<string, int> factor_sizes;  // factor_name -> size
};
```

### 3.2 Must Replicate (Algorithm Logic)

```cpp
// Factor-based propagation (from propagation.py)
struct FactorSharding {
    vector<string> axes;
    int priority;
    bool is_open;
};

// The Collect-Merge-Update phases
void propagate_sharding(OpShardingRule& rule,
                        vector<ShardingSpec>& input_specs,
                        vector<ShardingSpec>& output_specs);
```

### 3.3 Solver Interface (Input/Output)

```cpp
// INPUT to solver (from graph_extractor.py format)
struct GraphNode {
    string op_name;
    vector<int> input_ids;
    vector<int> output_ids;
    OpShardingRule rule;      // einsum-like equation
    float flops;              // from cost_model()
};

struct TensorInfo {
    int id;
    vector<int> shape;
    optional<ShardingSpec> constraint;  // Fixed sharding (input/output)
};

// OUTPUT from solver
struct NodeSolution {
    map<int, ShardingSpec> inputs;   // Required input sharding
    map<int, ShardingSpec> outputs;  // Resulting output sharding
};
```

---

## 4. Factor-Based Propagation for View Ops

### 4.1 Reshape Example (from `view.py`)

```python
# Input: [100, 20] -> Output: [2000]
# Input has atomic factors: d0=100, d1=20
# Output maps to compound: (d0 d1) = 2000

# Sharding rule: "d0 d1 -> (d0 d1)"
# If d0 is sharded on "x", output inherits sharding on "x"
# because d0 contributes to the compound dimension
```

### 4.2 Concat Example (from `view.py`)

```python
# All inputs share factor "c_concat" on concat axis
# This enforces: if ANY input is sharded on concat axis,
# ALL inputs and output must use same sharding

# Rule: "c_concat d1, c_concat d1, ... -> c_concat d1"
```

### 4.3 Gather/Scatter (Index Operations)

```python
# Data[d0, d1, ...], Indices[i0, i1, ...] -> Output[d_prefix, i0, i1, d_suffix]
# Indices factors REPLACE gathered dimension's factor
```

**Key insight**: Nabla correctly tracks data flow through complex ops via factors. This is **better** than GSPMD's dimension heuristics.

---

## 5. What ALPA ILP Would Look Like for Nabla

### 5.1 Decision Variables
```
s_{v,f,a} ∈ {0,1}  // Factor f of node v sharded on axis a
```

Instead of ALPA's per-dimension `s_vi`, we need per-**factor** decisions.

### 5.2 Constraints

**Factor consistency**: All tensors sharing a factor must agree on sharding.
```
∀ factor f: s_{v1,f,a} = s_{v2,f,a}  // for all tensors using factor f
```

**Axis exclusivity**: Each axis shards at most one factor per tensor.
```
∀ tensor t, axis a: Σ_f s_{t,f,a} ≤ 1
```

### 5.3 Objective

Same structure as ALPA but with factor-level costs:
```
min Σ (compute_cost + comm_cost) + Σ resharding_cost
```

---

## 6. Recommended Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python (Nabla)                        │
├─────────────────────────────────────────────────────────┤
│  trace() → graph_extractor → JSON Graph                 │
│                    ↓                                     │
│            ┌──────────────────┐                          │
│            │   Solver (C++)   │ ← ILP/DP engine          │
│            │  - Factor specs  │                          │
│            │  - Cost model    │                          │
│            │  - Propagation   │ (replicated from Python) │
│            └──────────────────┘                          │
│                    ↓                                     │
│            JSON Solution (per-node specs)                │
│                    ↓                                     │
│  shard_map replay with node constraints                  │
└─────────────────────────────────────────────────────────┘
```

**Current interface is already suitable**: JSON graph in, JSON solution out.

---

## 7. Action Items to Enable ALPA-like Solver

### Priority 1: Communication Cost Model
- Add `collective_cost(op, spec, mesh)` function
- Model AllReduce, AllGather, ReduceScatter costs
- Add resharding cost estimation

### Priority 2: Document Factor Semantics
- Formalize factor algebra for solver implementers
- Document how factors map to dimensions per op type

### Priority 3: Memory Estimation
- Add per-tensor memory size to graph
- Add activation memory tracking for pipeline scheduling

### NOT Needed Now:
- Solver interface abstraction (C++ solver can use current JSON format)
- Python base classes (solver is independent)
