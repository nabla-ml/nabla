# Rigorous Specification of the Alpa Algorithm

## 1. Core Definitions & Problem Space
* **Goal:** Map a computational graph $G$ (ordered operators $o_1 \dots o_K$) to a Cluster Mesh $\mathcal{C}$ of physical shape $(N, M)$.
* **Hierarchy:**
    * **Inter-Op Level:** Slices $G$ into stages and assigns them to device submeshes.
    * **Intra-Op Level:** Partitions operators within a stage onto the devices of a single submesh using SPMD.
* **Pipeline Schedule:** Synchronous 1F1B (One-Forward-One-Backward), chosen for memory efficiency.

---

## 2. Phase 1: Operator Clustering (Graph Compression)
**Objective:** Reduce the graph from thousands of operators to a sequence of layers $l_1 \dots l_L$ ($L \ll K$) to make the Inter-Op DP tractable.

**Cost Function $C(i, k)$:**
The total size (in bytes) of all tensors produced by operators $o_i \dots o_k$ that are consumed by operators outside this range (i.e., communication cost if cut).

**DP State $G(k, r)$:**
The minimum of the *maximal* incoming communication cost for any single layer, when clustering the first $k$ operators into $r$ layers.

**Recurrence (Eq. 6):**
$$
G(k, r) = \min_{1 \le i < k} \Big( \max \big( G(i, r-1), \ C(i+1, k) \big) \Big)
$$


**Validity Constraint:**
A split at $i$ is valid only if the FLOPs of the new layer satisfy the computational balance constraint:
$$
FLOP(i+1, k) \le (1 + \delta) \times \frac{FLOP_{total}}{L_{target}}
$$
Where $\delta$ is a tolerance parameter and ties are broken by minimizing FLOP variance.

---

## 3. Phase 2: Intra-Operator Parallelism (The ILP)
**Context:** Solved for a specific stage (subgraph of layers) and a specific **Logical** Mesh shape $(n, m)$.

**Sets & Constants:**
* $S_v$: Set of valid parallel algorithms (sharding specs) for node $v$.
* $c_{vi}$: Communication cost for node $v$ using spec $i$.
* $d_{vi}$: Compute cost for node $v$ using spec $i$ (simplified to 0 for heavy ops as work is divided evenly).
* $R_{vuij}$: Resharding cost for edge $(v,u)$ switching from spec $i$ to $j$.

**Decision Variables:**
* $s_{vi} \in \{0, 1\}$: Node $v$ selects spec $i$.
* $e_{vuij} \in \{0, 1\}$: Edge $(v,u)$ transitions from spec $i$ to $j$.

**Objective Function (Eq. 1):**
$$
\min \sum_{v \in V} \sum_{i} s_{vi} (c_{vi} + d_{vi}) + \sum_{(v,u) \in E} \sum_{i,j} e_{vuij} R_{vuij}
$$


**Constraints:**
1.  **Unique Selection:** $\sum_{i} s_{vi} = 1$ for all $v$.
2.  **Linearization:** To linearize the quadratic resharding term ($s_{vi} \cdot s_{uj}$), we enforce:
    $$\sum_{j} e_{vuij} = s_{vi} \quad \text{and} \quad \sum_{i} e_{vuij} = s_{uj}$$
    This ensures $e_{vuij}=1$ if and only if both connected nodes pick the corresponding specs.

**Output:** Minimum Latency $t_{intra}$, Parameter Memory $mem_{param}$, Activation Memory $mem_{act}$.

---

## 4. Phase 3: Inter-Operator Parallelism (The DP)
**Objective:** Minimize Pipeline Latency $T = \sum t_i + (B-1) \cdot t_{max}$.

**Valid Submesh Shapes:**
To ensure full cluster coverage, valid shapes are restricted to power-of-2 fractions of a row ($(1, 2^k)$) or full rows ($(k, M)$).

**DP State $F(s, k, d; t_{max})$:**
The minimal total latency to execute layers $k \dots L$ using exactly $s$ stages and $d$ devices, such that **no stage latency exceeds $t_{max}$**.

**Recurrence (Eq. 3):**
$$
F(s, k, d; t_{max}) = \min_{\substack{k < i \le L \\ \text{shape} \in \text{Valid}}} \Big\{ t_{intra}(k\dots i, \text{shape}) + F(s-1, i, d - \text{dev}(\text{shape}); t_{max}) \Big\}
$$


**Constraints for Validity (inside the Min):**
1.  **Latency:** $t_{intra}(\text{stage}) \le t_{max}$.
2.  **Memory (1F1B):** The stage must fit in device memory given the pipeline depth $s$:
    $$mem_{param} + s \cdot mem_{act} \le mem_{device}$$
    Note: $s$ represents the number of active microbatches/stages required for the schedule.

---

## 5. Pseudocode

```python
def Alpa_Rigorous(Graph G, Cluster (N, M), Microbatches B, Mem_Limit):
    
    # 1. Clustering
    Layers = OperatorClustering(G) 
    L = len(Layers)
    
    # 2. Pre-compute Intra-Op Costs (Memoization)
    # Runs the ILP for every valid sub-sequence of layers on every valid submesh.
    # CostCache key: (start_layer, end_layer, submesh_shape)
    CostCache = RunIntraOpPass(Layers, ValidShapes(N, M))
    
    # 3. Inter-Op DP
    # Get all unique latencies from profiling to use as t_max candidates
    Candidates = sorted(unique([c.latency for c in CostCache.values()]))
    GlobalMin = Infinity

    # Iterate t_max (bottleneck latency)
    for t_max in Candidates:
        # Optimization: Early Pruning
        if (B - 1) * t_max >= GlobalMin: break

        # F[stages_left][layer_idx][devices_left]
        F = array(size=(L+1, L+1, N*M+1), fill=Infinity)
        F[0][L][0] = 0  # Base case: 0 stages for L layers using 0 devices

        # s: Total number of stages in the current pipeline hypothesis
        for s in range(1, L + 1):
            
            # k: Current starting layer index (working backwards)
            for k in range(L - 1, -1, -1):
                
                # d: Devices available
                for d in range(1, N*M + 1):
                    
                    # i: Try splitting stage from k to i
                    for i in range(k + 1, L + 1):
                        for shape in ValidShapes:
                            n_dev = shape.n * shape.m
                            if n_dev > d: continue
                            
                            if (k, i, shape) not in CostCache: continue
                            (lat, m_param, m_act) = CostCache[(k, i, shape)]
                            
                            # --- CONSTRAINTS ---
                            # 1. Latency Bottleneck
                            if lat > t_max: continue
                            
                            # 2. Memory Constraint (Rigorous 1F1B)
                            # We use 's' here because it represents the pipeline depth
                            if m_param + s * m_act > Mem_Limit: continue
                            
                            # Recurrence
                            if F[s-1][i][d - n_dev] != Infinity:
                                cost = lat + F[s-1][i][d - n_dev]
                                F[s][k][d] = min(F[s][k][d], cost)

        # Calculate Total Pipeline Latency for this t_max
        # We check all pipeline depths 's' that consume ALL devices (d=N*M)
        for s in range(1, L + 1):
            if F[s][0][N*M] != Infinity:
                total_latency = F[s][0][N*M] + (B - 1) * t_max
                GlobalMin = min(GlobalMin, total_latency)

    return GlobalMin

```
---

## 6. Runtime Optimization: Cross-Mesh Resharding
**Problem:** Communicating tensors between Stage $A$ (Mesh 1) and Stage $B$ (Mesh 2) where meshes have different shapes or sharding specs.

**Algorithm (Local All-Gather):**
1.  **Check Replication:** Identify if the destination tensor spec on Mesh 2 involves replication.
2.  **Rewrite Communication:**
    * Instead of naive many-to-many send/recv (which uses slow cross-mesh links), send the tensor slice to only *one* device per replica group in Mesh 2.
    * **Local Broadcast:** The devices in Mesh 2 use `all-gather` (fast intra-mesh bandwidth) to replicate the data locally.