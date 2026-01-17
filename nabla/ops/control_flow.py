# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any, Callable, Sequence, TYPE_CHECKING
import functools

from max import graph
from max.graph import ops

from .operation import Operation, ensure_tensor
from ..core import pytree
# from ..core.tensor import Tensor # Moved to local imports to avoid cycle
from ..core import TensorImpl
from ..core import GRAPH

if TYPE_CHECKING:
    from ..core.tensor import Tensor
    from ..sharding.spec import DeviceMesh

if TYPE_CHECKING:
    from ..sharding.spec import DeviceMesh


def _unwrap_tensor(x: Any) -> Any:
    """Unwrap Tensor to TensorValue for MAX ops."""
    from ..core.tensor import Tensor
    if isinstance(x, Tensor):
        if not x._impl._values:
             # Ensure realized or lazy value exists
             # For control flow inside maxpr, we expect _values to be populated (shards)
             pass
        # Return the first value if it's a list (implicit single shard or replicated)
        # TODO: Handle multi-shard control flow properly (requires vector control flow or pmap)
        if hasattr(x, '_impl') and x._impl._values:
             return x._impl._values[0] 
        return x
    return x

def _wrap_tensor(x: Any, like: Tensor | None = None) -> Tensor:
    """Wrap TensorValue from MAX op back to Tensor."""
    if isinstance(x, (graph.TensorValue, graph.BufferValue)):
         # In eager/simulation mode, we wrapp result in a new Tensor
         # sharding etc will be inferred or set later
         from ..core.tensor import Tensor
         return Tensor(values=[x])
    return x # Already wrapped or other type


class WhereOp(Operation):
    """Element-wise conditional selection: where(cond, x, y).
    
    For each element: output[i] = x[i] if cond[i] else y[i]
    All inputs must be broadcast-compatible.
    """
    
    @property
    def name(self) -> str:
        return "where"
    
    def maxpr(
        self, 
        condition: graph.TensorValue, 
        x: graph.TensorValue, 
        y: graph.TensorValue
    ) -> graph.TensorValue:
        return ops.where(condition, x, y)
    
    # sharding_rule inherited from Operation (elementwise)




class CondOp(Operation):
    @property
    def name(self) -> str:
        return "cond"
    
    def __call__(
        self, 
        pred: Tensor | bool, 
        true_fn: Callable[..., Any], 
        false_fn: Callable[..., Any], 
        *operands: Any
    ) -> Any:
        from ..core.tensor import Tensor
        operands = pytree.tree_map(ensure_tensor, operands)
        if not isinstance(pred, Tensor):
             pred = ensure_tensor(pred)
        return super().__call__(pred, true_fn, false_fn, *operands)

    def infer_sharding_spec(self, args: tuple, mesh: "DeviceMesh", kwargs: dict = None):
        """Cond: Output sharding is determined by operands/branches.
        
        For now, we enforce that outputs inherit sharding from operands if they match structure.
        Ideally we would trace true_fn/false_fn.
        """
        # Args: pred, true_fn, false_fn, *operands
        # We assume result matches operands sharding? Not necessarily.
        # Fallback to default propagation (inherit from first sharded input) is risky if pred is sharded.
        return None, [], False
        
    def maxpr(self, pred_shard, true_fn, false_fn, *operand_shards):
        # Helper to trace function and get return types
        def wrapped_fn(fn, input_tensors):
            return fn(*input_tensors)
            
        from ..core.tensor import Tensor
        wrapped_operand_shards = pytree.tree_map(lambda x: Tensor(value=x), operand_shards)
        
        def max_true_fn():
            res = true_fn(*wrapped_operand_shards)
            return pytree.tree_map(_unwrap_tensor, res)
            
        def max_false_fn():
            res = false_fn(*wrapped_operand_shards)
            return pytree.tree_map(_unwrap_tensor, res)
            
        # Determine out_types by tracing true_fn in a scratch graph
        from max.graph import Graph
        out_types = []
        with Graph("scratch"):
             scratch_res = max_true_fn()
             def extract_type(x):
                 return x.type
             out_types = pytree.tree_map(extract_type, scratch_res)
             
        flat_out_types = pytree.tree_leaves(out_types)
        
        def flat_max_true_fn():
            r = max_true_fn()
            return pytree.tree_leaves(r)
            
        def flat_max_false_fn():
            r = max_false_fn()
            return pytree.tree_leaves(r)
            
        res_flat = ops.cond(pred_shard, flat_out_types, flat_max_true_fn, flat_max_false_fn)
        
        return pytree.tree_unflatten(pytree.tree_structure(scratch_res), res_flat)



class WhileLoopOp(Operation):
    @property
    def name(self) -> str:
        return "while_loop"
    
    def __call__(self, cond_fn: Callable, body_fn: Callable, init_val: Any) -> Any:
        from ..core.tensor import Tensor
        from ..core import TensorImpl

        from ..core import pytree
        from ..sharding import spmd
        from max import graph as g

        # 1. Collect inputs (init_val structure)
        # cond_fn, body_fn are callables, not Tensors. init_val is the data.
        args = (cond_fn, body_fn, init_val) # For consistency with maxpr structure
        
        # Collect metadata from init_val
        leaves = pytree.tree_leaves(init_val)
        any_traced = any(x._impl.traced for x in leaves if isinstance(x, Tensor))
        max_batch_dims = max((x._impl.batch_dims for x in leaves if isinstance(x, Tensor)), default=0)
        any_sharded = any(x._impl.is_sharded for x in leaves if isinstance(x, Tensor))
        
        # 2. Determine execution mode
        mesh = spmd.get_mesh_from_args(leaves) if any_sharded else None
        
        # 3. Setup Specs
        # WhileLoop Invariant: Output Spec == Input Spec
        # We need per-leaf specs.
        leaf_specs = []
        for x in leaves:
            if isinstance(x, Tensor) and x._impl.sharding:
                 leaf_specs.append(x._impl.sharding)
            else:
                 # If mesh exists, default to Replicated Open
                 if mesh:
                      rank = len(x.shape) if isinstance(x, Tensor) else 0
                      # Use PHYSICAL rank? 
                      if isinstance(x, Tensor):
                          rank = len(x.shape) + x._impl.batch_dims # Physical rank?
                          # But if not sharded, batch_dims might be 0 or simulated.
                          # Use x.global_shape?
                          # Safest: Create replicated spec.
                          from ..sharding.spmd import create_replicated_spec
                          leaf_specs.append(create_replicated_spec(mesh, rank))
                      else:
                          leaf_specs.append(None)
                 else:
                      leaf_specs.append(None)

        # 4. Reshard Inputs?

        # 5. Execute Loop
        num_shards = len(mesh.devices) if mesh else 1
        shard_results = []
        
        with GRAPH.graph:
            for shard_idx in range(num_shards):
                shard_init_val = spmd.get_shard_args(
                    init_val, shard_idx, leaf_specs, g, Tensor, pytree
                )
                res = self.maxpr(cond_fn, body_fn, shard_init_val) # Pass structure
                shard_results.append(res)
                
        # 6. Reconstruct Outputs (per leaf)
        if not shard_results:
             return None
             
        # Result structure should match init_val
        # Flatten results
        flat_results_per_shard = [pytree.tree_leaves(res) for res in shard_results]
        treedef = pytree.tree_structure(shard_results[0])
        num_leaves = len(flat_results_per_shard[0])
        
        if num_leaves != len(leaf_specs):
             # Should match if body_fn preserves structure (required by while_loop)
             pass
             
        output_leaves = []
        for i in range(num_leaves):
            leaf_shards = [shard[i] for shard in flat_results_per_shard]
            spec = leaf_specs[i] if i < len(leaf_specs) else None
            
            tensor = spmd.create_sharded_output(
                 leaf_shards, spec, any_traced, max_batch_dims, mesh=mesh
            )
            output_leaves.append(tensor)
            
        return pytree.tree_unflatten(treedef, output_leaves)

    def infer_sharding_spec(self, args: tuple, mesh: "DeviceMesh", kwargs: dict = None):
        """Unused by custom __call__ but kept for interface compliance."""
        return None, [], False


    def maxpr(self, cond_fn, body_fn, *init_shards):
        init_val_shard = init_shards[0]
        
        from ..core.tensor import Tensor
        def wrap(x): return Tensor(value=x)
        def unwrap(x): return x._impl._values[0] if isinstance(x, Tensor) else x
        
        def max_cond_fn(*args_flat):
            args_struct = pytree.tree_unflatten(pytree.tree_structure(init_val_shard), args_flat)
            wrapped_args = pytree.tree_map(wrap, args_struct)
            res = cond_fn(wrapped_args)
            return unwrap(res)

        def max_body_fn(*args_flat):
             args_struct = pytree.tree_unflatten(pytree.tree_structure(init_val_shard), args_flat)
             wrapped_args = pytree.tree_map(wrap, args_struct)
             res = body_fn(wrapped_args)
             return pytree.tree_leaves(pytree.tree_map(unwrap, res))

        init_flat = pytree.tree_leaves(init_val_shard)
        res_flat = ops.while_loop(init_flat, max_cond_fn, max_body_fn)
        
        return pytree.tree_unflatten(pytree.tree_structure(init_val_shard), res_flat)


# Singletons
_where_op = WhereOp()
_cond_op = CondOp()
_while_loop_op = WhileLoopOp()

# Public API

def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return _where_op(condition, x, y)

def cond(pred: Tensor, true_fn: Callable, false_fn: Callable, *operands: Any) -> Any:
    return _cond_op(pred, true_fn, false_fn, *operands)

def while_loop(cond_fn: Callable, body_fn: Callable, init_val: Any) -> Any:
    return _while_loop_op(cond_fn, body_fn, init_val)


# =============================================================================
# Scan Operation - Minimal Implementation for PP
# =============================================================================

def scan(
    f: Callable, 
    init: Any, 
    xs: Any, 
    length: int | None = None, 
    reverse: bool = False
) -> tuple[Any, Any]:
    """Scan implementation using while_loop.
    
    IMPORTANT: This is a minimal implementation for Pipeline Parallelism.
    For sharded tensors, ensure the scan dimension is NOT sharded.
    
    Args:
        f: Function (carry, x) -> (carry, y)
        init: Initial carry value
        xs: Inputs with leading dimension being scanned
        length: Scan length (optional, inferred from xs)
        reverse: If True, scan in reverse order
        
    Returns:
        (final_carry, stacked_outputs)
    """
    if reverse:
        raise NotImplementedError("Reverse scan not implemented yet")
    
    # 1. Infer length from xs
    xs_flat = pytree.tree_leaves(xs)
    if not xs_flat:
        raise ValueError("scan requires non-empty xs")
    
    first_x = xs_flat[0]
    inferred_length = int(first_x.shape[0])
    
    if length is not None and length != inferred_length:
        raise ValueError(f"Explicit length {length} != inferred length {inferred_length}")
    length = inferred_length
    
    if length == 0:
        raise NotImplementedError("Zero-length scan not implemented")
    
    # 2. For MVP: Use Python loop instead of while_loop
    # This is simpler and works correctly with sharding
    # The performance cost is acceptable for PP demos
    from ..ops import view, creation
    
    carry = init
    ys_list = []
    
    for i in range(length):
        # Slice xs at index i using static slicing
        def _slice_at_i(x):
            # Use local_shape for size since slice_tensor operates on physical data
            local_shape = x.local_shape
            start = [i] + [0] * (x.rank - 1)
            size = [1] + [int(d) for d in local_shape[1:]]
            slc = view.slice_tensor(x, start=start, size=size)
            return view.squeeze(slc, 0)
        
        x_i = pytree.tree_map(_slice_at_i, xs)
        
        # Apply user function
        carry, y_i = f(carry, x_i)
        ys_list.append(y_i)
    
    # 3. Stack outputs along axis 0
    def _stack_outputs(ys_leaves_list):
        # ys_leaves_list is list of leaves across iterations
        # Stack them to form (length, ...) tensor
        return view.stack(ys_leaves_list, axis=0)
    
    # Handle pytree structure of outputs
    if ys_list:
        # Get structure from first output
        first_y_flat = pytree.tree_leaves(ys_list[0])
        treedef = pytree.tree_structure(ys_list[0])
        num_leaves = len(first_y_flat)
        
        # Transpose: list of trees -> tree of lists
        stacked_leaves = []
        for leaf_idx in range(num_leaves):
            leaves_for_this_idx = [pytree.tree_leaves(y)[leaf_idx] for y in ys_list]
            stacked = _stack_outputs(leaves_for_this_idx)
            stacked_leaves.append(stacked)
        
        stacked_ys = pytree.tree_unflatten(treedef, stacked_leaves)
    else:
        stacked_ys = None
    
    return carry, stacked_ys


__all__ = ["where", "cond", "while_loop", "scan"]
