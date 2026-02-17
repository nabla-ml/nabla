# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..core import Trace, pytree, trace
from ..core.tensor import Tensor, TensorImpl
from ..ops.base import Operation

if TYPE_CHECKING:
    from ..core.graph.tracing import Trace
    from ..core.sharding.spec import DeviceMesh, ShardingSpec


def shard_map(
    func: Callable[..., Any],
    mesh: DeviceMesh,
    in_specs: dict[int, ShardingSpec],
    out_specs: dict[int, ShardingSpec] | None = None,
    auto_sharding: bool = False,
    debug: bool = False,
) -> Callable[..., Any]:
    """Execute a function with automatic sharding propagation and execution."""

    def _resolve_dual(x: Any) -> Any:
        from ..core import TensorImpl

        if isinstance(x, Tensor):
            return x.dual if x.dual is not None else x
        if isinstance(x, TensorImpl):
            return Tensor(impl=x.dual) if x.dual is not None else Tensor(impl=x)
        return x

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logical_args = list(args)
        for i, val in enumerate(logical_args):
            if isinstance(val, Tensor) and i in in_specs:
                spec = in_specs[i]
                if spec is not None:
                    logical_args[i] = val.shard(mesh, spec.dim_specs)

        traced = trace(func, *logical_args, **kwargs)

        if auto_sharding:
            from ..core.sharding.optimizer.auto_sharding import AutoSharding
            from ..core.sharding.spec import DimSpec

            extractor = _ShardingGraphExtractor(
                traced, in_specs, out_specs, debug=debug
            )
            json_graph = extractor.extract()
            solver = AutoSharding(mesh.shape, mesh.axis_names)
            solution = solver.solve(json_graph, debug=debug)

        pytree.tree_map(
            lambda x: x.realize() if isinstance(x, Tensor) else x, (args, kwargs)
        )

        try:
            flat_args, _ = pytree.tree_flatten(args)
            for i, x in enumerate(flat_args):
                if isinstance(x, Tensor):
                    spec = in_specs.get(i)
                    if spec:
                        from ..core.sharding.spec import needs_reshard

                        if x.sharding and needs_reshard(x.sharding, spec):
                            import warnings

                            warnings.warn(
                                f"shard_map input {i} is already sharded with {x.sharding} "
                                f"but in_spec wants {spec}. Using existing sharding. "
                                f"To force a specific sharding, pass an unsharded tensor.",
                                UserWarning,
                            )
                            x.dual = x
                        elif spec is not None:
                            x.dual = x.shard(mesh, spec.dim_specs)
                        else:
                            x.dual = x
                    else:
                        x.dual = x

            for node_idx, refs in enumerate(traced.nodes):
                if not refs.op_args:
                    if debug and auto_sharding:
                        print(
                            f"[shard_map] WARNING: Skipping node {node_idx} (no op_args - constant subgraph)"
                        )
                    continue

                dual_args = pytree.tree_map(_resolve_dual, refs.op_args)

                node_key = str(node_idx)
                has_solution = (
                    auto_sharding
                    and "nodes" in solution
                    and node_key in solution["nodes"]
                )

                if auto_sharding and not has_solution and debug:
                    print(
                        f"[shard_map] WARNING: No solution for node {node_idx} [{refs.op.name}]"
                    )

                if has_solution:
                    node_sol = solution["nodes"][str(node_idx)]
                    input_constraints = node_sol.get("inputs", {})

                    flat_duals, tree_def = pytree.tree_flatten(dual_args)
                    for i, d in enumerate(flat_duals):
                        if str(i) in input_constraints and isinstance(d, Tensor):
                            spec_dict = input_constraints[str(i)]

                            dim_specs = []
                            for ax in spec_dict["dims"]:
                                if ax is None:
                                    dim_specs.append(DimSpec([], is_open=True))
                                else:
                                    dim_specs.append(DimSpec(ax, is_open=True))

                            flat_duals[i] = d.shard(mesh, dim_specs)

                    dual_args = pytree.tree_unflatten(tree_def, flat_duals)

                dual_kwargs = (
                    pytree.tree_map(_resolve_dual, refs.op_kwargs)
                    if refs.op_kwargs
                    else {}
                )

                if debug and auto_sharding:
                    flat_check, _ = pytree.tree_flatten(dual_args)
                    input_shardings = []
                    for d in flat_check:
                        if isinstance(d, Tensor) and hasattr(d, "_impl") and d.sharding:
                            input_shardings.append(str(d.sharding))
                        else:
                            input_shardings.append("unsharded")
                    print(
                        f"[shard_map] Node {node_idx} [{refs.op.name}] INPUT shardings: {input_shardings}"
                    )
                    if "nodes" in solution and str(node_idx) in solution["nodes"]:
                        print(
                            f"[shard_map]   Solver wanted inputs: {solution['nodes'][str(node_idx)].get('inputs', {})}"
                        )

                result = refs.op(*dual_args, **dual_kwargs)

                if debug and auto_sharding:
                    flat_res_check = [
                        x for x in pytree.tree_leaves(result) if isinstance(x, Tensor)
                    ]
                    output_shardings = []
                    for r in flat_res_check:
                        if hasattr(r, "_impl") and r.sharding:
                            output_shardings.append(str(r.sharding))
                        else:
                            output_shardings.append("unsharded")
                    print(
                        f"[shard_map] Node {node_idx} [{refs.op.name}] OUTPUT shardings (eager): {output_shardings}"
                    )
                    if "nodes" in solution and str(node_idx) in solution["nodes"]:
                        print(
                            f"[shard_map]   Solver wanted outputs: {solution['nodes'][str(node_idx)].get('outputs', {})}"
                        )

                flat_res = [
                    x for x in pytree.tree_leaves(result) if isinstance(x, Tensor)
                ]
                alive_outs = [o for o in refs.get_alive_outputs() if o is not None]

                if flat_res:
                    for i, (logical, physical) in enumerate(
                        zip(alive_outs, flat_res, strict=False)
                    ):
                        if (
                            auto_sharding
                            and "nodes" in solution
                            and str(node_idx) in solution["nodes"]
                        ):
                            node_sol = solution["nodes"][str(node_idx)]
                            out_constraints = node_sol.get("outputs", {})

                            if str(i) in out_constraints:
                                spec_dict = out_constraints[str(i)]
                                dim_specs = []
                                for ax in spec_dict["dims"]:
                                    if ax is None:
                                        dim_specs.append(DimSpec([], is_open=True))
                                    else:
                                        dim_specs.append(DimSpec(ax, is_open=True))
                                physical = physical.shard(mesh, dim_specs)

                        logical.dual = physical._impl

            res = pytree.tree_map(
                lambda x: x.dual if isinstance(x, Tensor) and x.dual is not None else x,
                traced.outputs,
            )

            if out_specs:
                flat_outs, tree = pytree.tree_flatten(res)
                for i, val in enumerate(flat_outs):
                    if (
                        i in out_specs
                        and isinstance(val, Tensor)
                        and out_specs[i] is not None
                    ):
                        flat_outs[i] = val.shard(mesh, out_specs[i].dim_specs)
                res = pytree.tree_unflatten(tree, flat_outs)

            return res

        finally:
            inputs = pytree.tree_leaves((args, kwargs))
            nodes = traced.nodes if traced._computed else []

            for x in inputs:
                if isinstance(x, Tensor):
                    x.dual = None

            for refs in nodes:
                for out in refs.get_alive_outputs():
                    if out:
                        out.dual = None

    return wrapper


class _ShardingGraphExtractor:
    """Extracts a JSON graph representation from a logical trace."""

    def __init__(
        self,
        trace: Trace,
        in_specs: dict[int, ShardingSpec],
        out_specs: dict[int, ShardingSpec] | None = None,
        debug: bool = False,
    ) -> None:
        self.trace = trace
        self.in_specs = in_specs
        self.out_specs = out_specs or {}
        self.debug = debug

        self.tensors: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.tensor_id_map: dict[int, int] = {}
        self.id_to_tensor_impl: dict[int, TensorImpl] = {}
        self.next_tensor_id = 0

    def _get_or_create_tensor_id(self, tensor_impl: TensorImpl) -> int:
        if id(tensor_impl) not in self.tensor_id_map:
            json_id = self.next_tensor_id
            self.next_tensor_id += 1
            self.tensor_id_map[id(tensor_impl)] = json_id
            self.id_to_tensor_impl[json_id] = tensor_impl

            shape = tensor_impl.global_shape
            if shape is None:
                shape = tensor_impl.physical_shape

            shape_tuple = tuple(int(d) for d in shape) if shape else ()

            fixed_sharding = None
            if tensor_impl.sharding_constraint:
                spec = tensor_impl.sharding_constraint
                fixed_sharding = {
                    "dims": [d.axes if d.axes else None for d in spec.dim_specs],
                    "replicated": list(spec.replicated_axes),
                }

            self.tensors.append(
                {
                    "id": json_id,
                    "shape": shape_tuple,
                    "dtype": str(tensor_impl.dtype) if tensor_impl.dtype else "float32",
                    "size_bytes": 0,
                    "fixed_sharding": fixed_sharding,
                }
            )

        return self.tensor_id_map[id(tensor_impl)]

    def extract(self) -> str:
        """Run extraction and return JSON string."""
        if not self.trace._computed:
            self.trace.compute()

        flat_args, _ = pytree.tree_flatten(self.trace.inputs)
        for i, val in enumerate(flat_args):
            if isinstance(val, Tensor):
                tid = self._get_or_create_tensor_id(val._impl)
                if i in self.in_specs:
                    spec = self.in_specs[i]
                    if spec:
                        self.tensors[tid]["fixed_sharding"] = {
                            "dims": [
                                d.axes if d.axes else None for d in spec.dim_specs
                            ],
                            "replicated": list(spec.replicated_axes),
                        }

        for i, refs in enumerate(self.trace.nodes):
            op: Operation = refs.op

            input_ids = []
            input_shapes = []

            def collect_inputs(x, input_ids=input_ids, input_shapes=input_shapes):
                from ..core import TensorImpl

                if isinstance(x, Tensor):
                    input_ids.append(self._get_or_create_tensor_id(x._impl))
                    input_shapes.append(tuple(int(d) for d in x.shape))
                elif isinstance(x, TensorImpl):
                    input_ids.append(self._get_or_create_tensor_id(x))
                    input_shapes.append(
                        tuple(int(d) for d in x.global_shape or x.physical_shape)
                    )

            pytree.tree_map(collect_inputs, refs.op_args)
            if refs.op_kwargs:
                pytree.tree_map(collect_inputs, refs.op_kwargs)

            output_ids = []
            output_shapes = []
            outputs = refs.get_alive_outputs()
            valid_outputs = [o for o in outputs if o is not None]

            for out in valid_outputs:
                output_ids.append(self._get_or_create_tensor_id(out))
                output_shapes.append(
                    tuple(int(d) for d in out.global_shape or out.physical_shape)
                )

            rule_info = None
            try:
                rule = op.sharding_rule(
                    input_shapes, output_shapes, **(refs.op_kwargs or {})
                )
                if rule:
                    rule_info = {
                        "equation": rule.to_einsum_notation(),
                        "factor_sizes": rule.factor_sizes,
                    }
            except Exception as e:
                if self.debug:
                    print(
                        f"[GraphExtractor] WARNING: Failed to get sharding_rule for node {i} [{op.name}]: {e}"
                    )

            cost = op.compute_cost(input_shapes, output_shapes)

            self.nodes.append(
                {
                    "id": i,
                    "op_name": op.name,
                    "inputs": input_ids,
                    "outputs": output_ids,
                    "sharding_rule": rule_info,
                    "compute_stats": {"flops": cost},
                }
            )

        flat_outs, _ = pytree.tree_flatten(self.trace.outputs)
        for i, val in enumerate(flat_outs):
            if i in self.out_specs and isinstance(val, Tensor):
                tid = self._get_or_create_tensor_id(val._impl)
                spec = self.out_specs[i]
                self.tensors[tid]["fixed_sharding"] = {
                    "dims": [d.axes if d.axes else None for d in spec.dim_specs],
                    "replicated": list(spec.replicated_axes),
                }

        graph_data = {
            "meta": {"mesh": {}},
            "tensors": self.tensors,
            "nodes": self.nodes,
        }

        json_output = json.dumps(graph_data, indent=2)

        if self.debug:
            print("\\n[AutoSharding] Extracted Graph JSON:")
            print(json_output)
            print("-" * 50)

        return json_output
