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
) -> Callable[..., Any]:
    """Execute a function with automatic sharding propagation and execution.

    Args:
        func: The function to transform.
        mesh: The device mesh to use.
        in_specs: Input sharding constraints (index -> spec).
        out_specs: Output sharding constraints (index -> spec).
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

        # Pre-realize inputs preventing lazy graph pollution
        pytree.tree_map(lambda x: x._sync_realize() if isinstance(x, Tensor) else x, (args, kwargs))

        try:
            # Attach input duals
            flat_args, _ = pytree.tree_flatten(args)
            for i, x in enumerate(flat_args):
                if isinstance(x, Tensor):
                    spec = in_specs.get(i)
                    x.dual = x.shard(mesh, spec.dim_specs) if spec else x

            # Replay graph on duals
            for refs in traced.nodes:
                if not refs.op_args: continue # Skip untraced constant subgraphs

                dual_args = pytree.tree_map(_resolve_dual, refs.op_args)
                dual_kwargs = pytree.tree_map(_resolve_dual, refs.op_kwargs) if refs.op_kwargs else {}
                
                result = refs.op(*dual_args, **dual_kwargs)
                
                # Update output duals
                flat_res = [x for x in pytree.tree_leaves(result) if isinstance(x, Tensor)]
                alive_outs = [o for o in refs.get_alive_outputs() if o is not None]
                
                if flat_res:
                    for logical, physical in zip(alive_outs, flat_res):
                        if logical.sharding_constraint:
                            physical = physical.shard(
                                logical.sharding_constraint.mesh,
                                logical.sharding_constraint.dim_specs
                            )
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
