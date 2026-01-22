# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Core backpropagation utilities for Trace-based automatic differentiation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph.tracing import Trace
    from ..tensor.impl import TensorImpl


def backward_on_trace(
    trace: Trace,
    cotangents: Any,
    *,
    checkpoint_policy: str = "none",
) -> dict[int, TensorImpl]:
    """Pure-function backpropagation on a Trace.
    
    Args:
        trace: Captured computation graph with inputs and outputs
        cotangents: PyTree of cotangent tensors matching trace.outputs structure
        checkpoint_policy: Rematerialization strategy ("none" | "checkpoint" | "recompute_all")
    
    Returns:
        Mapping from input TensorImpl id to gradient TensorImpl
    
    Algorithm:
        1. Realize all trace inputs (populate _storages)
        2. Rehydrate trace (repopulate all TensorImpl._values)
        3. Initialize output cotangents
        4. Backward traversal: compute VJPs in reverse topological order
        5. Return gradients for trace inputs
    """
    from ..common import pytree
    from ..tensor.api import Tensor
    from ..tensor.impl import TensorImpl
    from ..graph.engine import GRAPH
    
    # Ensure trace is computed
    if not trace._computed:
        trace.compute()
    
    # Step 1 & 2: Realize inputs and rehydrate the trace
    input_leaves = [
        t for t in pytree.tree_leaves(trace.inputs) if isinstance(t, Tensor)
    ]
    
    # Realize all inputs
    for inp in input_leaves:
        if not inp._impl.is_realized:
            GRAPH.evaluate(inp)
    
    # Rehydrate the trace (populates all TensorImpl._values)
    trace.rehydrate()
    
    # Step 3: Initialize cotangents for outputs
    output_leaves = [
        t._impl for t in pytree.tree_leaves(trace.outputs) if isinstance(t, Tensor)
    ]
    
    cotangent_leaves = [
        t._impl for t in pytree.tree_leaves(cotangents) if isinstance(t, Tensor)
    ]
    
    if len(cotangent_leaves) != len(output_leaves):
        raise ValueError(
            f"Number of cotangents ({len(cotangent_leaves)}) must match "
            f"number of outputs ({len(output_leaves)})"
        )
    
    # Initialize cotangent storage - map from TensorImpl id to cotangent TensorImpl
    cotangent_map: dict[int, TensorImpl] = {}
    
    for output_impl, cotangent_impl in zip(output_leaves, cotangent_leaves, strict=True):
        cotangent_map[id(output_impl)] = cotangent_impl
    
    # Step 4: Backward Traversal - compute VJPs in reverse topological order
    for output_refs in reversed(trace.nodes):
        # Get alive outputs
        alive_outputs = output_refs.get_alive_outputs()
        
        # Check if any output has a cotangent
        has_cotangent = False
        for out_impl in alive_outputs:
            if out_impl is not None and id(out_impl) in cotangent_map:
                has_cotangent = True
                break
        
        if not has_cotangent:
            continue
        
        # Get the operation
        op = output_refs.op
        
        # Check if operation has VJP rule
        if not hasattr(op, 'vjp_rule'):
            continue
        
        # Reconstruct primals (inputs) as Tensors
        primals_as_tensors = []
        for arg in pytree.tree_leaves(output_refs.op_args):
            if isinstance(arg, TensorImpl):
                primals_as_tensors.append(Tensor(impl=arg))
            elif isinstance(arg, Tensor):
                primals_as_tensors.append(arg)
            else:
                # Non-tensor argument (scalar, etc.)
                primals_as_tensors.append(arg)
        
        # Reconstruct with original structure
        primals_structured = pytree.tree_unflatten(
            pytree.tree_structure(output_refs.op_args),
            primals_as_tensors
        )
        
        # Reconstruct outputs as Tensors
        output_tensors = []
        for out_impl in alive_outputs:
            if out_impl is not None:
                output_tensors.append(Tensor(impl=out_impl))
            else:
                output_tensors.append(None)
        
        outputs_structured = pytree.tree_unflatten(
            output_refs.tree_def, output_tensors
        )
        
        # Get cotangents for outputs (create zeros if missing)
        output_cotangents = []
        for out_impl in alive_outputs:
            if out_impl is not None and id(out_impl) in cotangent_map:
                output_cotangents.append(Tensor(impl=cotangent_map[id(out_impl)]))
            else:
                # Create zero cotangent
                from ...ops.creation import zeros_like
                output_cotangents.append(zeros_like(Tensor(impl=out_impl)))
        
        cotangents_structured = pytree.tree_unflatten(
            output_refs.tree_def, output_cotangents
        )
        
        # Convert to scalar if single output
        if len(output_cotangents) == 1:
            cotangents_structured = cotangents_structured[0] if isinstance(cotangents_structured, (list, tuple)) else cotangents_structured
            outputs_structured = outputs_structured[0] if isinstance(outputs_structured, (list, tuple)) else outputs_structured
        
        # Call VJP rule
        try:
            input_cotangents = op.vjp_rule(
                primals_structured,
                cotangents_structured,
                outputs_structured
            )
        except Exception as e:
            op_name = getattr(op, 'name', str(op))
            raise RuntimeError(
                f"VJP rule failed for operation '{op_name}': {e}"
            ) from e
        
        # Accumulate cotangents for inputs
        arg_leaves = [
            arg._impl if isinstance(arg, Tensor) else arg 
            for arg in pytree.tree_leaves(output_refs.op_args)
        ]
        
        cotangent_result_leaves = [
            ct._impl if isinstance(ct, Tensor) else ct
            for ct in pytree.tree_leaves(input_cotangents)
        ]
        
        for arg_impl, cotangent_impl in zip(arg_leaves, cotangent_result_leaves, strict=False):
            if not isinstance(arg_impl, TensorImpl):
                continue
            
            arg_id = id(arg_impl)
            
            if arg_id in cotangent_map:
                # Accumulate: existing + new
                from ...ops.binary import add
                existing = Tensor(impl=cotangent_map[arg_id])
                new = Tensor(impl=cotangent_impl) if isinstance(cotangent_impl, TensorImpl) else cotangent_impl
                accumulated = add(existing, new)
                cotangent_map[arg_id] = accumulated._impl
            else:
                # First cotangent for this input
                cotangent_map[arg_id] = cotangent_impl if isinstance(cotangent_impl, TensorImpl) else cotangent_impl._impl
    
    # Step 5: Collect gradients for trace inputs
    gradients = {}
    for inp in input_leaves:
        inp_id = id(inp._impl)
        if inp_id in cotangent_map:
            gradients[inp_id] = cotangent_map[inp_id]
        else:
            # No gradient computed - return zeros
            from ...ops.creation import zeros_like
            zero_grad = zeros_like(inp)
            gradients[inp_id] = zero_grad._impl
    
    # Step 6: Cleanup - clear cotangents from all TensorImpls in the trace
    cotangent_map.clear()
    
    return gradients


__all__ = ["backward_on_trace"]
