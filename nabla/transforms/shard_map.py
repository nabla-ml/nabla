# ===----------------------------------------------------------------------=== #
# Nabla 2026
# ===----------------------------------------------------------------------=== #

"""shard_map: Automatic sharding propagation and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from ..core import pytree
from ..core.tensor import Tensor
from ..core.tensor_impl import TensorImpl
from ..core.trace import trace

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
            from .graph_extractor import ShardingGraphExtractor
            from ..optimizer.simple_solver import SimpleSolver
            from ..sharding.spec import ShardingSpec, DimSpec

            # 1. Extract
            extractor = ShardingGraphExtractor(traced, in_specs, out_specs, debug=debug)
            json_graph = extractor.extract()

            # 2. Solve
            solver = SimpleSolver(mesh.shape, mesh.axis_names)
            solution = solver.solve(json_graph, debug=debug)

            # 3. NO-OP: Applying constraints to Tensors globally is removed (mostly).
            # Instead we apply them per-node during replay.
            # But we might want to capture "Default" tensor states if we wanted global coherence?
            # For now, let's rely on Node Inputs forcing the state.

        # Pre-realize inputs preventing lazy graph pollution
        pytree.tree_map(lambda x: x._sync_realize() if isinstance(x, Tensor) else x, (args, kwargs))

        try:
            # Attach input duals matches in_specs (Boundary condition)
            flat_args, _ = pytree.tree_flatten(args)
            for i, x in enumerate(flat_args):
                if isinstance(x, Tensor):
                    spec = in_specs.get(i)
                    x.dual = x.shard(mesh, spec.dim_specs) if spec else x

            # Replay graph on duals
            for node_idx, refs in enumerate(traced.nodes):
                if not refs.op_args: continue # Skip untraced constant subgraphs

                dual_args = pytree.tree_map(_resolve_dual, refs.op_args)
                
                # AUTO SHARDING: Apply Input Constraints for this Node
                if auto_sharding and "nodes" in solution and str(node_idx) in solution["nodes"]:
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
                
                result = refs.op(*dual_args, **dual_kwargs)
                
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
