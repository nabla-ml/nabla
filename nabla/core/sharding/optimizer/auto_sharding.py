# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..propagation import (
    OpShardingRuleTemplate,
    PropagationStrategy,
    propagate_sharding,
)
from ..spec import DeviceMesh, DimSpec, ShardingSpec


class AutoSharding:
    """Solver that uses factor-based propagation with cost-based seeding.

    Phases:
    1. Parse: JSON -> ShardingSpec
    2. Seed: Cost-based heuristics (e.g. Matmul DP vs MP)
    3. Propagate: Bidirectional flow
    4. Export: Specs -> JSON
    """

    def __init__(self, mesh_shape: tuple[int, ...], axis_names: tuple[str, ...]):
        self.mesh_shape = mesh_shape
        self.axis_names = axis_names
        self.mesh = DeviceMesh(
            name="solver_mesh",
            shape=mesh_shape,
            axis_names=axis_names,
            devices=list(range(self._compute_total_devices(mesh_shape))),
        )
        self.total_devices = self._compute_total_devices(mesh_shape)

    @staticmethod
    def _compute_total_devices(shape: tuple[int, ...]) -> int:
        total = 1
        for d in shape:
            total *= d
        return total

    def solve(self, json_graph: str, debug: bool = False) -> dict[str, Any]:
        """Solve for optimal sharding specs using propagation."""
        if debug:
            print("\n[AutoSharding] Starting Propagation-Based Solver...")

        graph = json.loads(json_graph)
        tensors = {t["id"]: t for t in graph["tensors"]}
        nodes = graph["nodes"]

        tensor_specs: dict[int, ShardingSpec] = {}
        for t_id, t_info in tensors.items():
            shape = tuple(t_info["shape"])
            spec = self._create_initial_spec(shape, t_info.get("fixed_sharding"))
            tensor_specs[t_id] = spec

        if debug:
            print(f"[Solver] Created {len(tensor_specs)} tensor specs")

        seeded_nodes: dict[str, dict] = {}
        for node in nodes:
            op_name = node["op_name"]
            node_id = str(node["id"])

            if debug:
                print(f"[Solver] Analyzing Node {node_id}: {op_name}")

            if op_name == "matmul":
                seeding = self._seed_matmul(node, tensors, tensor_specs, debug)
                if seeding:
                    seeded_nodes[node_id] = seeding

        MAX_ITERATIONS = 100
        for iteration in range(MAX_ITERATIONS):
            changed = False
            for node in nodes:
                if self._propagate_node(node, tensors, tensor_specs, debug):
                    changed = True

            if debug:
                print(
                    f"[Solver] Propagation iteration {iteration + 1}: changed={changed}"
                )

            if not changed:
                if debug:
                    print(
                        f"[Solver] Fixed-point reached after {iteration + 1} iterations"
                    )
                break
        else:
            if debug:
                print(
                    f"[Solver] WARNING: Did not converge after {MAX_ITERATIONS} iterations"
                )

        solution = self._export_solution(
            nodes, tensors, tensor_specs, seeded_nodes, debug
        )

        if debug:
            print(f"[Solver] Final Solution:\n{json.dumps(solution, indent=2)}")
            print("-" * 50)

        return solution

    def _create_initial_spec(
        self, shape: tuple[int, ...], fixed: dict | None
    ) -> ShardingSpec:
        """Create initial ShardingSpec for a tensor."""
        if fixed:
            dim_specs = []
            for ax in fixed.get("dims", []):
                if ax is None:
                    dim_specs.append(DimSpec([], is_open=True))
                else:
                    dim_specs.append(DimSpec(ax, is_open=False))
            replicated = frozenset(fixed.get("replicated", []))
        else:
            dim_specs = [DimSpec([], is_open=True) for _ in shape]
            replicated = frozenset()

        return ShardingSpec(
            mesh=self.mesh, dim_specs=dim_specs, replicated_axes=replicated
        )

    def _seed_matmul(
        self,
        node: dict,
        tensors: dict,
        tensor_specs: dict[int, ShardingSpec],
        debug: bool = False,
    ) -> dict | None:
        """Cost-based seeding for matmul: choose DP vs MP."""
        from nabla.ops.communication import AllReduceOp

        if TYPE_CHECKING:
            pass

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

        dp_cost = float("inf")
        if m_size % mesh_dim == 0:
            dp_compute = flops / mesh_dim
            dp_comm = 0.0
            dp_cost = dp_compute + dp_comm

        mp_cost = float("inf")
        if k_size % mesh_dim == 0:
            mp_compute = flops / mesh_dim

            shape_out = tuple(tensors[node["outputs"][0]]["shape"])

            num_elements = 1
            for d in shape_out:
                num_elements *= d
            output_bytes = num_elements * 4

            mp_comm = AllReduceOp.estimate_cost(output_bytes, self.mesh, [axis_name])

            mp_cost = mp_compute + mp_comm

        if debug:
            print(
                f"  > Costs: DP={dp_cost:.2e}, MP={mp_cost:.2e} (M={m_size}, K={k_size}, N={n_size})"
            )

        in_a_id = node["inputs"][0]
        in_b_id = node["inputs"][1]
        out_id = node["outputs"][0]

        _rank_a = len(tensors[in_a_id]["shape"])
        _rank_b = len(tensors[in_b_id]["shape"])
        _out_rank = len(tensors[out_id]["shape"])

        strategy = "none"

        if dp_cost < mp_cost and dp_cost != float("inf"):
            strategy = "dp"
            if debug:
                print("  > Selected Strategy: Data Parallel (Split M)")

            self._set_dim_sharding(tensor_specs[in_a_id], -2, [axis_name])
            self._set_dim_sharding(tensor_specs[out_id], -2, [axis_name])

        elif mp_cost != float("inf"):
            strategy = "mp"
            if debug:
                print("  > Selected Strategy: Model Parallel (Split K)")

            self._set_dim_sharding(tensor_specs[in_a_id], -1, [axis_name])
            self._set_dim_sharding(tensor_specs[in_b_id], -2, [axis_name])

        return {"strategy": strategy, "axis": axis_name}

    def _set_dim_sharding(
        self, spec: ShardingSpec, dim_idx: int, axes: list[str]
    ) -> None:
        """Set sharding for a dimension (handles negative indices)."""
        if dim_idx < 0:
            dim_idx = len(spec.dim_specs) + dim_idx
        if 0 <= dim_idx < len(spec.dim_specs):
            spec.dim_specs[dim_idx] = DimSpec(axes=axes, is_open=True, priority=5)

    def _propagate_node(
        self,
        node: dict,
        tensors: dict,
        tensor_specs: dict[int, ShardingSpec],
        debug: bool = False,
    ) -> bool:
        """Propagate sharding through a single node (returns True if changed)."""
        rule_info = node.get("sharding_rule")
        if not rule_info or "equation" not in rule_info:
            return False

        input_shapes = [tuple(tensors[t_id]["shape"]) for t_id in node["inputs"]]
        output_shapes = [tuple(tensors[t_id]["shape"]) for t_id in node["outputs"]]

        equation = rule_info["equation"]

        try:
            template = OpShardingRuleTemplate.parse(equation, input_shapes)
            rule = template.instantiate(input_shapes, output_shapes)
        except Exception as e:
            if debug:
                print(
                    f"[Solver] WARNING: Failed to parse rule for node {node['id']} [{node['op_name']}]: {e}"
                )
                print(f"         Equation: {equation}")
            return False

        input_specs = [tensor_specs[t_id] for t_id in node["inputs"]]
        output_specs = [tensor_specs[t_id] for t_id in node["outputs"]]

        changed = propagate_sharding(
            rule, input_specs, output_specs, strategy=PropagationStrategy.BASIC
        )
        return changed

    def _export_solution(
        self,
        nodes: list[dict],
        tensors: dict,
        tensor_specs: dict[int, ShardingSpec],
        seeded_nodes: dict[str, dict],
        debug: bool = False,
    ) -> dict[str, Any]:
        """Export propagated specs to node-centric solution format."""
        solution = {"nodes": {}}

        for node in nodes:
            node_id = str(node["id"])

            node_sol = {"inputs": {}, "outputs": {}}

            for i, t_id in enumerate(node["inputs"]):
                spec = tensor_specs[t_id]
                node_sol["inputs"][str(i)] = self._spec_to_dict(spec)

            for i, t_id in enumerate(node["outputs"]):
                spec = tensor_specs[t_id]
                node_sol["outputs"][str(i)] = self._spec_to_dict(spec)

            solution["nodes"][node_id] = node_sol

        return solution

    def _spec_to_dict(self, spec: ShardingSpec) -> dict[str, Any]:
        """Convert ShardingSpec to dict format for JSON."""
        dims = []
        for ds in spec.dim_specs:
            if ds.axes:
                dims.append(list(ds.axes))
            else:
                dims.append(None)
        return {"dims": dims, "replicated": list(spec.replicated_axes)}
