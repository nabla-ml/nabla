# ===----------------------------------------------------------------------=== #
# Nabla 2026
# ===----------------------------------------------------------------------=== #

"""Simple Solver: Optimizes sharding strategies based on cost model."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

class SimpleSolver:
    """Greedy solver that selects the best sharding strategy for each node."""

    def __init__(self, mesh_shape: Tuple[int, ...], axis_names: Tuple[str, ...]):
        self.mesh_shape = mesh_shape
        self.axis_names = axis_names
        self.total_devices = 1
        for d in mesh_shape:
            self.total_devices *= d

    def solve(self, json_graph: str, debug: bool = False) -> Dict[int, Any]:
        """Solve for optimal sharding specs.
        
        Returns:
            Dict mapping tensor_id -> ShardingSpec dict representation
            {
                "tensor_id": { "dims": [["x"], None], "replicated": [] }
            }
        """
        if debug:
            print("\n[AutoSharding] Starting Solver...")

        graph = json.loads(json_graph)
        tensors = {t["id"]: t for t in graph["tensors"]}
        nodes = graph["nodes"]
        
        solution = {}
        
        # Greedy node-by-node optimization
        # In a real solver, we would propagate costs globally (DP or ILP).
        # Here we just look at each op and pick the best local strategy.
        
        # We need to track assigned sharding to ensure consistency?
        # For now, let's just optimize independent ops and assume the propagation engine
        # will handle the resharding logic/costs between them if they mismatch.
        # But wait, if we pick A -> B mismatching, we pay cost. 
        # The simple solver described in the plan:
        # "Select strategy with Min Cost (Greedy for now)."
        
        for node in nodes:
            op_name = node["op_name"]
            
            if debug:
                print(f"[Solver] Analyzing Node {node['id']}: {op_name}")
            
            # DEFAULT: Replicated (Cost = Base Compute)
            # Todo: If inputs are already sharded, we might prefer preserving that?
            
            if op_name == "matmul":
                self._solve_matmul(node, tensors, solution, debug)
            else:
                # Default: Pass through or Replicated
                pass
                
        if debug:
            print(f"[Solver] Solution:\n{json.dumps(solution, indent=2)}")
            print("-" * 50)

        return solution

    def _solve_matmul(self, node: Dict, tensors: Dict, solution: Dict, debug: bool = False):
        """Optimize Matmul Strategy."""
        # Rule: mk,kn->mn
        # Strategies:
        # 1. Data Parallel (Split M): mk on x, kn repl -> mn on x. 
        #    Compute = Base/N. Comm = 0 (if inputs are right).
        # 2. Model Parallel (Split K): mk on x, kn on x -> mn partial -> AllReduce.
        #    Compute = Base/N. Comm = AllReduce(MN).
        
        rule = node.get("sharding_rule")
        if not rule: return
        
        factor_sizes = rule.get("factor_sizes", {})
        
        # Extract shapes
        # We assume standard dot product for simplicity of this prototype
        # factors: m, k, n
        
        m_size = factor_sizes.get("m", 1024)
        k_size = factor_sizes.get("k", 1024)
        n_size = factor_sizes.get("n", 1024)
        
        flops = node["compute_stats"]["flops"]
        
        # Cost Model Constants (Arbitrary for demo)
        TIME_PER_FLOP = 1e-12  # 1 TFLOPs device = 1e-12 s/flop? No 1e-12 is 1ps. 
        # Say 1 GFLOPS device. 1e-9 s/flop.
        # Let's say 1 unit of work.
        
        # Strategy 1: Data Parallel (Split M)
        # We need to check if M is divisible by mesh size
        # Let's assume 1D mesh for simplicity "data" axis
        axis_name = self.axis_names[0] if self.axis_names else "d"
        mesh_dim = self.mesh_shape[0] if self.mesh_shape else 1
        
        dp_cost = float("inf")
        if m_size % mesh_dim == 0:
            # Parallel compute
            compute = flops / mesh_dim
            # Comm cost: 0 (assuming inputs available)
            dp_cost = compute
            
        # Strategy 2: Model Parallel (Split K)
        mp_cost = float("inf")
        if k_size % mesh_dim == 0:
            compute = flops / mesh_dim
            # Comm cost: AllReduce(Output Size)
            # Output is M * N * 4 bytes
            comm_bytes = m_size * n_size * 4
            # Bandwidth 10 GB/s -> 1e10 bytes/s
            comm_time = comm_bytes / 1e10 
            # We need to normalize units. Let's assume flops is dominant usually?
            # 2e9 flops vs 4MB comm. 
            # 2e9 * 1e-12 = 0.002s. 
            # 4e6 / 1e10 = 0.0004s.
            # So MP is viable.
            
            # Using a simplified weight
            COMM_PENALTY_WEIGHT = 1000.0 # Emphasize comm cost
            mp_cost = compute + (comm_bytes * COMM_PENALTY_WEIGHT)

        if debug:
            print(f"  > Costs: DP={dp_cost:.2e}, MP={mp_cost:.2e} (M={m_size}, K={k_size}, N={n_size})")

        # Decision
        solution.setdefault("nodes", {})
             
        # Matmul output (generic logic from before but now structured)
        # Out ID
        out_id = node["outputs"][0]
        out_rank = len(tensors[out_id]["shape"])
             
        # Inputs
        in_a_id = node["inputs"][0]
        in_b_id = node["inputs"][1]
        rank_a = len(tensors[in_a_id]["shape"])
        rank_b = len(tensors[in_b_id]["shape"])

        # Prepare spec structures
        node_solution = {
            "inputs": {},
            "outputs": {}
        }
             
        if dp_cost < mp_cost and dp_cost != float("inf"):
            if debug:
                print("  > Selected Strategy: Data Parallel (Split M)")
                 
            # DP Specs:
            # Output: (..., m, n) -> Split M (dim -2)
            dims_out = [None] * out_rank
            dims_out[-2] = [axis_name]
                 
            # Input A: (..., m, k) -> Split M (dim -2)
            dims_a = [None] * rank_a
            dims_a[-2] = [axis_name]
                 
            # Input B: (..., k, n) -> Replicated
            dims_b = [None] * rank_b
                 
            node_solution["outputs"]["0"] = {"dims": dims_out, "replicated": []}
            node_solution["inputs"]["0"] = {"dims": dims_a, "replicated": []}
            node_solution["inputs"]["1"] = {"dims": dims_b, "replicated": []}
                 
        elif mp_cost != float("inf"):
            if debug:
                print("  > Selected Strategy: Model Parallel (Split K)")
                 
            # MP Specs:
            # Output: (..., m, n) -> Replicated
            dims_out = [None] * out_rank
                 
            # Input A: (..., m, k) -> Split K (dim -1)
            dims_a = [None] * rank_a
            dims_a[-1] = [axis_name]
            
            # Input B: (..., k, n) -> Split K (dim -2)
            dims_b = [None] * rank_b
            dims_b[-2] = [axis_name]
                 
            node_solution["outputs"]["0"] = {"dims": dims_out, "replicated": []}
            node_solution["inputs"]["0"] = {"dims": dims_a, "replicated": []}
            node_solution["inputs"]["1"] = {"dims": dims_b, "replicated": []}
             
        # Store in main solution
        solution["nodes"][str(node["id"])] = node_solution
