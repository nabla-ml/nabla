# ===----------------------------------------------------------------------=== #
# Nabla 2026
# ===----------------------------------------------------------------------=== #

"""Simple Solver: Optimizes sharding strategies using propagation infrastructure.

This solver integrates with the factor-based propagation system in propagation.py
to enable bidirectional sharding flow. It uses cost heuristics to seed initial
constraints, then propagates them through the graph.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from ..sharding.propagation import (
    OpShardingRule,
    OpShardingRuleTemplate,
    propagate_sharding,
    PropagationStrategy,
)


class SimpleSolver:
    """Solver that uses factor-based propagation with cost-based seeding.
    
    The solver works in three phases:
    1. Parse: Convert JSON graph into ShardingSpec objects
    2. Seed: Apply cost-based heuristics to set initial constraints (e.g., matmul DP/MP)
    3. Propagate: Use bidirectional propagation to flow constraints to all tensors
    4. Export: Convert propagated specs to node-centric solution JSON
    """

    def __init__(self, mesh_shape: Tuple[int, ...], axis_names: Tuple[str, ...]):
        self.mesh_shape = mesh_shape
        self.axis_names = axis_names
        self.mesh = DeviceMesh(
            name="solver_mesh",
            shape=mesh_shape,
            axis_names=axis_names,
            devices=list(range(self._compute_total_devices(mesh_shape)))
        )
        self.total_devices = self._compute_total_devices(mesh_shape)
    
    @staticmethod
    def _compute_total_devices(shape: Tuple[int, ...]) -> int:
        total = 1
        for d in shape:
            total *= d
        return total

    def solve(self, json_graph: str, debug: bool = False) -> Dict[str, Any]:
        """Solve for optimal sharding specs using propagation.
        
        Returns:
            Node-centric solution dict:
            {
                "nodes": {
                    "node_id": {
                        "inputs": { "0": {"dims": [...], "replicated": []} },
                        "outputs": { "0": {"dims": [...], "replicated": []} }
                    }
                }
            }
        """
        if debug:
            print("\n[AutoSharding] Starting Propagation-Based Solver...")

        graph = json.loads(json_graph)
        tensors = {t["id"]: t for t in graph["tensors"]}
        nodes = graph["nodes"]
        
        # Phase 1: Create ShardingSpecs for all tensors
        tensor_specs: Dict[int, ShardingSpec] = {}
        for t_id, t_info in tensors.items():
            shape = tuple(t_info["shape"])
            spec = self._create_initial_spec(shape, t_info.get("fixed_sharding"))
            tensor_specs[t_id] = spec
        
        if debug:
            print(f"[Solver] Created {len(tensor_specs)} tensor specs")
        
        # Phase 2: Cost-based seeding for key operations
        # This sets initial constraints that propagation will spread
        seeded_nodes: Dict[str, Dict] = {}
        for node in nodes:
            op_name = node["op_name"]
            node_id = str(node["id"])
            
            if debug:
                print(f"[Solver] Analyzing Node {node_id}: {op_name}")
            
            if op_name == "matmul":
                seeding = self._seed_matmul(node, tensors, tensor_specs, debug)
                if seeding:
                    seeded_nodes[node_id] = seeding
        
        # Phase 3: FIXED-POINT propagation across entire graph
        # Iterate until no changes, enabling true bidirectional flow
        MAX_ITERATIONS = 100
        for iteration in range(MAX_ITERATIONS):
            changed = False
            for node in nodes:
                if self._propagate_node(node, tensors, tensor_specs, debug):
                    changed = True
            
            if debug:
                print(f"[Solver] Propagation iteration {iteration + 1}: changed={changed}")
            
            if not changed:
                if debug:
                    print(f"[Solver] Fixed-point reached after {iteration + 1} iterations")
                break
        else:
            if debug:
                print(f"[Solver] WARNING: Did not converge after {MAX_ITERATIONS} iterations")
        
        # Phase 4: Export to node-centric solution format
        solution = self._export_solution(nodes, tensors, tensor_specs, seeded_nodes, debug)
        
        if debug:
            print(f"[Solver] Final Solution:\n{json.dumps(solution, indent=2)}")
            print("-" * 50)

        return solution
    
    def _create_initial_spec(
        self, shape: Tuple[int, ...], fixed: Optional[Dict]
    ) -> ShardingSpec:
        """Create initial ShardingSpec for a tensor."""
        if fixed:
            # Parse fixed constraint
            dim_specs = []
            for ax in fixed.get("dims", []):
                if ax is None:
                    dim_specs.append(DimSpec([], is_open=True))
                else:
                    dim_specs.append(DimSpec(ax, is_open=False))
            replicated = frozenset(fixed.get("replicated", []))
        else:
            # Default: fully open (can be sharded by propagation)
            dim_specs = [DimSpec([], is_open=True) for _ in shape]
            replicated = frozenset()
        
        return ShardingSpec(
            mesh=self.mesh,
            dim_specs=dim_specs,
            replicated_axes=replicated
        )
    
    def _seed_matmul(
        self,
        node: Dict,
        tensors: Dict,
        tensor_specs: Dict[int, ShardingSpec],
        debug: bool = False
    ) -> Optional[Dict]:
        """Cost-based seeding for matmul: choose DP vs MP.
        
        Uses communication cost model for accurate DP vs MP tradeoffs:
        - DP (Data Parallel): Split M dimension, no AllReduce needed
        - MP (Model Parallel): Split K dimension, requires AllReduce on output
        """
        from ..ops.binary import matmul
        from ..sharding.spec import ShardingSpec, DimSpec
        
        rule = node.get("sharding_rule")
        if not rule:
            return None
        
        factor_sizes = rule.get("factor_sizes", {})
        m_size = factor_sizes.get("m", 1024)
        k_size = factor_sizes.get("k", 1024)
        n_size = factor_sizes.get("n", 1024)
        
        flops = node["compute_stats"]["flops"]
        
        axis_name = self.axis_names[0] if self.axis_names else "d"
        mesh_dim = self.mesh_shape[0] if self.mesh_shape else 1
        
        # Cost calculation
        # DP: Split M dimension - no communication needed (each device has independent rows)
        dp_cost = float("inf")
        if m_size % mesh_dim == 0:
            dp_compute = flops / mesh_dim
            dp_comm = 0.0  # No AllReduce for non-contracting dimension
            dp_cost = dp_compute + dp_comm
        
        # MP: Split K dimension - requires AllReduce on output
        mp_cost = float("inf")
        if k_size % mesh_dim == 0:
            mp_compute = flops / mesh_dim
            
            # Create hypothetical specs for MP scenario to query cost
            # Input A: [..., M, K] -> shard K (-1)
            # Input B: [..., K, N] -> shard K (-2)
            # Output: [..., M, N] -> replicated (start with, but actually parallel execution produces partials)
            
            # NOTE: communication_cost expects global shapes
            shape_a = tuple(tensors[node["inputs"][0]]["shape"])
            shape_b = tuple(tensors[node["inputs"][1]]["shape"])
            shape_out = tuple(tensors[node["outputs"][0]]["shape"])
            
            # Construct dummy specs consistent with MP strategy
            # We only care about the sharded axes for cost calculation
            
            # A: Shard last dim (K)
            dims_a = [DimSpec([]) for _ in shape_a]
            dims_a[-1] = DimSpec([axis_name])
            spec_a = ShardingSpec(self.mesh, dims_a)
            
            # B: Shard second to last dim (K)
            dims_b = [DimSpec([]) for _ in shape_b]
            if len(dims_b) >= 2:
                dims_b[-2] = DimSpec([axis_name])
            spec_b = ShardingSpec(self.mesh, dims_b)
            
            # Output: Initally empty (sharding determined by op) 
            # or we pass what we expect? 
            # communication_cost check inputs to see if reduction is needed.
            # Output spec argument is currently unused in MatmulOp.communication_cost logic, 
            # but good to pass a placeholder.
            dims_out = [DimSpec([]) for _ in shape_out]
            spec_out = ShardingSpec(self.mesh, dims_out)
            
            # Query Op for cost
            mp_comm = matmul.communication_cost(
                [spec_a, spec_b], 
                [spec_out], 
                [shape_a, shape_b], 
                [shape_out], 
                self.mesh
            )
            
            mp_cost = mp_compute + mp_comm

        if debug:
            print(f"  > Costs: DP={dp_cost:.2e}, MP={mp_cost:.2e} (M={m_size}, K={k_size}, N={n_size})")

        # Apply seeding to tensor specs
        in_a_id = node["inputs"][0]
        in_b_id = node["inputs"][1]
        out_id = node["outputs"][0]
        
        rank_a = len(tensors[in_a_id]["shape"])
        rank_b = len(tensors[in_b_id]["shape"])
        out_rank = len(tensors[out_id]["shape"])
        
        strategy = "none"
        
        if dp_cost < mp_cost and dp_cost != float("inf"):
            strategy = "dp"
            if debug:
                print("  > Selected Strategy: Data Parallel (Split M)")
            
            # Seed constraints with low priority (will be propagated)
            self._set_dim_sharding(tensor_specs[in_a_id], -2, [axis_name])
            self._set_dim_sharding(tensor_specs[out_id], -2, [axis_name])
            # B stays replicated (open)
            
        elif mp_cost != float("inf"):
            strategy = "mp"
            if debug:
                print("  > Selected Strategy: Model Parallel (Split K)")
            
            self._set_dim_sharding(tensor_specs[in_a_id], -1, [axis_name])
            self._set_dim_sharding(tensor_specs[in_b_id], -2, [axis_name])
            # Output replicated (open)
        
        return {"strategy": strategy, "axis": axis_name}
    
    def _set_dim_sharding(
        self, spec: ShardingSpec, dim_idx: int, axes: List[str]
    ) -> None:
        """Set sharding for a dimension (handles negative indices)."""
        if dim_idx < 0:
            dim_idx = len(spec.dim_specs) + dim_idx
        if 0 <= dim_idx < len(spec.dim_specs):
            spec.dim_specs[dim_idx] = DimSpec(axes=axes, is_open=True, priority=5)
    
    def _propagate_node(
        self,
        node: Dict,
        tensors: Dict,
        tensor_specs: Dict[int, ShardingSpec],
        debug: bool = False
    ) -> bool:
        """Propagate sharding through a single node using factor-based propagation.
        
        Returns:
            True if any specs were modified, False otherwise.
        """
        rule_info = node.get("sharding_rule")
        if not rule_info or "equation" not in rule_info:
            return False
        
        # Get input/output shapes
        input_shapes = [tuple(tensors[t_id]["shape"]) for t_id in node["inputs"]]
        output_shapes = [tuple(tensors[t_id]["shape"]) for t_id in node["outputs"]]
        
        equation = rule_info["equation"]
        
        # Parse the sharding rule
        try:
            template = OpShardingRuleTemplate.parse(equation, input_shapes)
            rule = template.instantiate(input_shapes, output_shapes)
        except Exception as e:
            # If rule parsing fails, log warning and skip propagation for this node
            if debug:
                print(f"[Solver] WARNING: Failed to parse rule for node {node['id']} [{node['op_name']}]: {e}")
                print(f"         Equation: {equation}")
            return False
        
        # Collect specs
        input_specs = [tensor_specs[t_id] for t_id in node["inputs"]]
        output_specs = [tensor_specs[t_id] for t_id in node["outputs"]]
        
        # Propagate bidirectionally and return whether changes occurred
        changed = propagate_sharding(
            rule,
            input_specs,
            output_specs,
            strategy=PropagationStrategy.BASIC
        )
        return changed
    
    def _export_solution(
        self,
        nodes: List[Dict],
        tensors: Dict,
        tensor_specs: Dict[int, ShardingSpec],
        seeded_nodes: Dict[str, Dict],
        debug: bool = False
    ) -> Dict[str, Any]:
        """Export propagated specs to node-centric solution format."""
        solution = {"nodes": {}}
        
        for node in nodes:
            node_id = str(node["id"])
            
            # Build node solution
            node_sol = {
                "inputs": {},
                "outputs": {}
            }
            
            # Export input specs
            for i, t_id in enumerate(node["inputs"]):
                spec = tensor_specs[t_id]
                node_sol["inputs"][str(i)] = self._spec_to_dict(spec)
            
            # Export output specs
            for i, t_id in enumerate(node["outputs"]):
                spec = tensor_specs[t_id]
                node_sol["outputs"][str(i)] = self._spec_to_dict(spec)
            
            solution["nodes"][node_id] = node_sol
        
        return solution
    
    def _spec_to_dict(self, spec: ShardingSpec) -> Dict[str, Any]:
        """Convert ShardingSpec to dict format for JSON."""
        dims = []
        for ds in spec.dim_specs:
            if ds.axes:
                dims.append(list(ds.axes))
            else:
                dims.append(None)
        return {
            "dims": dims,
            "replicated": list(spec.replicated_axes)
        }
