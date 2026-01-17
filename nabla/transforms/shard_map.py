# ===----------------------------------------------------------------------=== #
# Nabla 2026
# ===----------------------------------------------------------------------=== #

"""shard_map: Automatic sharding propagation and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, List, Set, Tuple
import json

from ..core import pytree
from ..core.tensor import Tensor
from ..core import TensorImpl
from ..core import trace, Trace
from ..ops.operation import Operation

if TYPE_CHECKING:
    from ..sharding.mesh import DeviceMesh
    from ..sharding.spec import ShardingSpec


def shard_map(
    func: Callable[..., Any],
    mesh: DeviceMesh,
    in_specs: Dict[int, ShardingSpec],
    out_specs: Optional[Dict[int, ShardingSpec]] = None,
    auto_sharding: bool = False,
    debug: bool = False,
) -> Callable[..., Any]:
    """Execute a function with automatic sharding propagation and execution.

    Args:
        func: The function to transform.
        mesh: The device mesh to use.
        in_specs: Input sharding constraints (index -> spec).
        out_specs: Output sharding constraints (index -> spec).
        auto_sharding: If True, uses a solver to find optimal sharding constraints.
        debug: If True, prints detailed debug info during extraction and solving.
    """

    def _resolve_dual(x: Any) -> Any:
        # Helper to resolve logical tensor to its physical dual
        if isinstance(x, Tensor):
            return x.dual if x.dual is not None else x
        if isinstance(x, TensorImpl):
            return Tensor(impl=x.dual) if x.dual is not None else Tensor(impl=x)
        return x

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Capture logical graph
        traced = trace(func, *args, **kwargs)

        if auto_sharding:
            from ..sharding.optimizer.simple_solver import SimpleSolver
            from ..sharding.optimizer.simple_solver import SimpleSolver
            from ..sharding.spec import ShardingSpec, DimSpec

            # 1. Extract
            extractor = _ShardingGraphExtractor(traced, in_specs, out_specs, debug=debug)
            json_graph = extractor.extract()

            # 2. Solve
            solver = SimpleSolver(mesh.shape, mesh.axis_names)
            solution = solver.solve(json_graph, debug=debug)

            # 3. NO-OP: Applying constraints to Tensors globally is removed (mostly).
            # Instead we apply them per-node during replay.
            # But we might want to capture "Default" tensor states if we wanted global coherence?
            # For now, let's rely on Node Inputs forcing the state.

        # Pre-realize inputs preventing lazy graph pollution
        pytree.tree_map(lambda x: x.realize() if isinstance(x, Tensor) else x, (args, kwargs))

        try:
            # Attach input duals matches in_specs (Boundary condition)
            flat_args, _ = pytree.tree_flatten(args)
            for i, x in enumerate(flat_args):
                if isinstance(x, Tensor):
                    spec = in_specs.get(i)
                    if spec:
                        # Check if input is already sharded with a DIFFERENT spec
                        from ..sharding.spec import needs_reshard
                        if x._impl.sharding and needs_reshard(x._impl.sharding, spec):
                            # Pre-sharded with different spec - use existing sharding
                            # Resharding would cause shape mismatch with traced constants
                            import warnings
                            warnings.warn(
                                f"shard_map input {i} is already sharded with {x._impl.sharding} "
                                f"but in_spec wants {spec}. Using existing sharding. "
                                f"To force a specific sharding, pass an unsharded tensor.",
                                UserWarning
                            )
                            x.dual = x
                        else:
                            # Unsharded or same spec - apply in_spec
                            x.dual = x.shard(mesh, spec.dim_specs)
                    else:
                        x.dual = x

            # Replay graph on duals
            for node_idx, refs in enumerate(traced.nodes):
                if not refs.op_args:
                    if debug and auto_sharding:
                        print(f"[shard_map] WARNING: Skipping node {node_idx} (no op_args - constant subgraph)")
                    continue  # Skip untraced constant subgraphs

                dual_args = pytree.tree_map(_resolve_dual, refs.op_args)
                
                # AUTO SHARDING: Apply Input Constraints for this Node
                node_key = str(node_idx)
                has_solution = auto_sharding and "nodes" in solution and node_key in solution["nodes"]
                
                if auto_sharding and not has_solution and debug:
                    print(f"[shard_map] WARNING: No solution for node {node_idx} [{refs.op.name}]")
                
                if has_solution:
                    node_sol = solution["nodes"][str(node_idx)]
                    input_constraints = node_sol.get("inputs", {})
                    
                    # Flatten dual_args to apply by index
                    flat_duals, tree_def = pytree.tree_flatten(dual_args)
                    for i, d in enumerate(flat_duals):
                        if str(i) in input_constraints and isinstance(d, Tensor):
                            # Found a constraint!
                            spec_dict = input_constraints[str(i)]
                            
                            # Construct spec
                            dim_specs = []
                            for ax in spec_dict["dims"]:
                                if ax is None: dim_specs.append(DimSpec([], is_open=True))
                                else: dim_specs.append(DimSpec(ax, is_open=True))
                            
                            # Force Reshard (or Shard) to this spec
                            # logic: d.shard(mesh, dim_specs)
                            flat_duals[i] = d.shard(mesh, dim_specs)
                    
                    dual_args = pytree.tree_unflatten(tree_def, flat_duals)


                dual_kwargs = pytree.tree_map(_resolve_dual, refs.op_kwargs) if refs.op_kwargs else {}
                
                # DEBUG: Log sharding state before operation
                if debug and auto_sharding:
                    flat_check, _ = pytree.tree_flatten(dual_args)
                    input_shardings = []
                    for d in flat_check:
                        if isinstance(d, Tensor) and hasattr(d, '_impl') and d._impl.sharding:
                            input_shardings.append(str(d._impl.sharding))
                        else:
                            input_shardings.append("unsharded")
                    print(f"[shard_map] Node {node_idx} [{refs.op.name}] INPUT shardings: {input_shardings}")
                    if "nodes" in solution and str(node_idx) in solution["nodes"]:
                        print(f"[shard_map]   Solver wanted inputs: {solution['nodes'][str(node_idx)].get('inputs', {})}")
                
                result = refs.op(*dual_args, **dual_kwargs)
                
                # DEBUG: Log output sharding after operation
                if debug and auto_sharding:
                    flat_res_check = [x for x in pytree.tree_leaves(result) if isinstance(x, Tensor)]
                    output_shardings = []
                    for r in flat_res_check:
                        if hasattr(r, '_impl') and r._impl.sharding:
                            output_shardings.append(str(r._impl.sharding))
                        else:
                            output_shardings.append("unsharded")
                    print(f"[shard_map] Node {node_idx} [{refs.op.name}] OUTPUT shardings (eager): {output_shardings}")
                    if "nodes" in solution and str(node_idx) in solution["nodes"]:
                        print(f"[shard_map]   Solver wanted outputs: {solution['nodes'][str(node_idx)].get('outputs', {})}")
                
                # Update output duals
                flat_res = [x for x in pytree.tree_leaves(result) if isinstance(x, Tensor)]
                alive_outs = [o for o in refs.get_alive_outputs() if o is not None]
                
                if flat_res:
                    for i, (logical, physical) in enumerate(zip(alive_outs, flat_res)):
                        # If logical had a constraint?
                        # With Node-Centric, we might check 'outputs' in solution
                        if auto_sharding and "nodes" in solution and str(node_idx) in solution["nodes"]:
                             node_sol = solution["nodes"][str(node_idx)]
                             out_constraints = node_sol.get("outputs", {})
                             
                             # We assume alive_outs matches indices 0, 1... 
                             # This is naive if some outputs are dead/None. 
                             # Use 'out' index? simple_solver assumes '0' for first output.
                             # Let's trust that for single-output ops '0' is correct.
                             # For split: 0, 1.
                             if str(i) in out_constraints:
                                 spec_dict = out_constraints[str(i)]
                                 dim_specs = []
                                 for ax in spec_dict["dims"]:
                                     if ax is None: dim_specs.append(DimSpec([], is_open=True))
                                     else: dim_specs.append(DimSpec(ax, is_open=True))
                                 
                                 # Enforce output constraint?
                                 # Usually we don't 'cast' output, we just assert?
                                 # Or we cast? "shard" is a View if consistent.
                                 physical = physical.shard(mesh, dim_specs)
                        
                        logical.dual = physical._impl

            # Finalize outputs
            res = pytree.tree_map(lambda x: x.dual if isinstance(x, Tensor) and x.dual is not None else x, traced.outputs)
            
            if out_specs:
                flat_outs, tree = pytree.tree_flatten(res)
                for i, val in enumerate(flat_outs):
                    if i in out_specs and isinstance(val, Tensor) and out_specs[i]:
                        flat_outs[i] = val.shard(mesh, out_specs[i].dim_specs)
                res = pytree.tree_unflatten(tree, flat_outs)
            
            return res

        finally:
            # Cleanup dual references
            inputs = pytree.tree_leaves((args, kwargs))
            nodes = traced.nodes if traced._computed else []
            
            for x in inputs:
                if isinstance(x, Tensor): x.dual = None
            
            for refs in nodes:
                 for out in refs.get_alive_outputs():
                      if out: out.dual = None

    return wrapper


class _ShardingGraphExtractor:
    """Extracts a JSON graph representation from a logical trace for sharding optimization."""

    def __init__(self, trace: Trace, in_specs: Dict[int, Any], out_specs: Optional[Dict[int, Any]] = None, debug: bool = False):
        self.trace = trace
        self.in_specs = in_specs
        self.out_specs = out_specs or {}
        self.debug = debug
        
        self.tensors: List[Dict[str, Any]] = []
        self.nodes: List[Dict[str, Any]] = []
        self.tensor_id_map: Dict[int, int] = {}  # id(TensorImpl) -> json_id
        self.id_to_tensor_impl: Dict[int, TensorImpl] = {} # json_id -> TensorImpl
        self.next_tensor_id = 0

    def _get_or_create_tensor_id(self, tensor_impl: TensorImpl) -> int:
        if id(tensor_impl) not in self.tensor_id_map:
            json_id = self.next_tensor_id
            self.next_tensor_id += 1
            self.tensor_id_map[id(tensor_impl)] = json_id
            self.id_to_tensor_impl[json_id] = tensor_impl
            
            # Extract tensor metadata
            shape = tensor_impl.global_shape
            if shape is None:
                shape = tensor_impl.physical_shape
            
            shape_tuple = tuple(int(d) for d in shape) if shape else ()
            
            # Determine if this tensor has a fixed sharding constraint
            fixed_sharding = None
            if tensor_impl.sharding_constraint:
                spec = tensor_impl.sharding_constraint
                # Basic serialization of spec (we might need more detail later)
                fixed_sharding = {
                    "dims": [d.axes if d.axes else None for d in spec.dim_specs],
                    "replicated": list(spec.replicated_axes)
                }

            self.tensors.append({
                "id": json_id,
                "shape": shape_tuple,
                "dtype": str(tensor_impl.dtype) if tensor_impl.dtype else "float32",
                "size_bytes": 0,  # Placeholder
                "fixed_sharding": fixed_sharding
            })
            
        return self.tensor_id_map[id(tensor_impl)]

    def extract(self) -> str:
        """Run extraction and return JSON string."""
        if not self.trace._computed:
            self.trace.compute()

        # 1. Register Inputs
        # Flatten input args to match in_specs indices
        flat_args, _ = pytree.tree_flatten(self.trace.inputs)
        for i, val in enumerate(flat_args):
            if isinstance(val, Tensor):
                tid = self._get_or_create_tensor_id(val._impl)
                # Apply input constraint if provided
                if i in self.in_specs:
                    spec = self.in_specs[i]
                    if spec:
                        # Update the tensor definition with this constraint
                        # Note: trace inputs might already have constraints on them, 
                        # but in_specs passed to shard_map are the "boundary conditions".
                        # We merge or overwrite.
                        self.tensors[tid]["fixed_sharding"] = {
                            "dims": [d.axes if d.axes else None for d in spec.dim_specs],
                            "replicated": list(spec.replicated_axes)
                        }

        # 2. Process Nodes
        for i, refs in enumerate(self.trace.nodes):
            op: Operation = refs.op
            
            # Inputs
            input_ids = []
            input_shapes = []
            
            def collect_inputs(x):
                if isinstance(x, Tensor):
                    input_ids.append(self._get_or_create_tensor_id(x._impl))
                    input_shapes.append(tuple(int(d) for d in x.shape))
                elif isinstance(x, TensorImpl):
                    input_ids.append(self._get_or_create_tensor_id(x))
                    input_shapes.append(tuple(int(d) for d in x.global_shape or x.physical_shape))
            
            pytree.tree_map(collect_inputs, refs.op_args)
            if refs.op_kwargs:
                pytree.tree_map(collect_inputs, refs.op_kwargs)

            # Outputs
            output_ids = []
            output_shapes = []
            outputs = refs.get_alive_outputs()
            valid_outputs = [o for o in outputs if o is not None]
            
            for out in valid_outputs:
                output_ids.append(self._get_or_create_tensor_id(out))
                output_shapes.append(tuple(int(d) for d in out.global_shape or out.physical_shape))

            # Sharding Rule & Cost
            rule_info = None
            try:
                # Instantiate rule to get factors
                # Note: This might fail if rule generation is complex or args missing.
                # We assume simple rules for now.
                rule = op.sharding_rule(input_shapes, output_shapes, **(refs.op_kwargs or {}))
                if rule:
                    rule_info = {
                        "equation": rule.to_einsum_notation(),
                        "factor_sizes": rule.factor_sizes
                    }
            except Exception as e:
                # Log the error in debug mode
                if self.debug:
                    print(f"[GraphExtractor] WARNING: Failed to get sharding_rule for node {i} [{op.name}]: {e}")

            cost = op.compute_cost(input_shapes, output_shapes)

            self.nodes.append({
                "id": i,
                "op_name": op.name,
                "inputs": input_ids,
                "outputs": output_ids,
                "sharding_rule": rule_info,
                "compute_stats": {
                    "flops": cost
                }
            })
            
        # 3. Register Outputs (Constraints)
        flat_outs, _ = pytree.tree_flatten(self.trace.outputs)
        for i, val in enumerate(flat_outs):
             if i in self.out_specs and isinstance(val, Tensor):
                 tid = self._get_or_create_tensor_id(val._impl)
                 spec = self.out_specs[i]
                 # We need to enforce this constraint on the tensor produced by the last node
                 self.tensors[tid]["fixed_sharding"] = {
                    "dims": [d.axes if d.axes else None for d in spec.dim_specs],
                    "replicated": list(spec.replicated_axes)
                 }

        # Build final JSON
        graph_data = {
            "meta": {
                # Placeholder for mesh info - passed separately to solver usually, 
                # but good to have if we knew it here from in_specs[0].mesh
                "mesh": {} 
            },
            "tensors": self.tensors,
            "nodes": self.nodes
        }
        
        json_output = json.dumps(graph_data, indent=2)
        
        if self.debug:
            print("\\n[AutoSharding] Extracted Graph JSON:")
            print(json_output)
            print("-" * 50)
            
        return json_output
