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


def _unwrap_single(tree: Any) -> Any:
    """Unwrap a single-leaf tree if it's a list/tuple of size 1."""
    if isinstance(tree, (list, tuple)) and len(tree) == 1:
        return tree[0]
    return tree


def backward_on_trace(
    trace: Trace,
    cotangents: Any,
    *,
    create_graph: bool = False,
    checkpoint_policy: str = "none",
) -> dict[Tensor, Tensor]:
    """Pure-function backpropagation on a Trace.

    Args:
        trace: Captured computation graph with inputs and outputs
        cotangents: PyTree of cotangent tensors matching trace.outputs structure
        create_graph: Whether to trace the backward computations (enabling higher-order derivs)
        checkpoint_policy: Rematerialization strategy ("none" | "checkpoint" | "recompute_all")

    Returns:
        Mapping from input Tensor to gradient Tensor
    """
    from ..common import pytree
    from ..tensor.api import Tensor
    from ..tensor.impl import TensorImpl
    from ..graph.engine import GRAPH

    if not trace._computed:
        trace.compute()

    # Rehydrate internal graph values (this also ensures all leaves are realized)
    trace.rehydrate()

    # Prepare global cotangent map
    input_leaves = [
        t for t in pytree.tree_leaves(trace.inputs) if isinstance(t, Tensor)
    ]
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

    cotangent_map: dict[int, TensorImpl] = {}
    for output_impl, cotangent_impl in zip(
        output_leaves, cotangent_leaves, strict=True
    ):
        cotangent_map[id(output_impl)] = cotangent_impl

    # Helper to manage traced state during VJP execution
    original_flags: dict[int, bool] = {}

    def set_trace_state(obj: Any):
        if isinstance(obj, Tensor):
            original_flags[id(obj)] = obj.traced
            if not create_graph:
                obj.traced = False

    def restore_trace_state(obj: Any):
        if isinstance(obj, Tensor) and id(obj) in original_flags:
            obj.traced = original_flags[id(obj)]

    # Traverse captured nodes in reverse topological order
    for output_refs in reversed(trace.nodes):
        alive_outputs = output_refs.get_alive_outputs()

        # Check if any output of this operation has a known cotangent
        has_cotangent = False
        for out_impl in alive_outputs:
            if out_impl is not None and id(out_impl) in cotangent_map:
                has_cotangent = True
                break

        if not has_cotangent:
            continue

        op = output_refs.op
        if not hasattr(op, "vjp_rule"):
            continue

        # 1. Prepare Primals (inputs to the operation)
        primals_flat = []
        for arg in pytree.tree_leaves(output_refs.op_args):
            if isinstance(arg, TensorImpl):
                t = Tensor(impl=arg)
                primals_flat.append(t)
            elif isinstance(arg, Tensor):
                primals_flat.append(arg)
            else:
                primals_flat.append(arg)

        primals_structured = pytree.tree_unflatten(
            pytree.tree_structure(output_refs.op_args), primals_flat
        )

        # 2. Prepare Outputs (they carry the OutputRefs/op_kwargs)
        output_tensors = []
        for out_impl in alive_outputs:
            if out_impl is not None:
                t = Tensor(impl=out_impl)
                output_tensors.append(t)
            else:
                output_tensors.append(None)

        outputs_structured = pytree.tree_unflatten(output_refs.tree_def, output_tensors)

        # 3. Prepare Cotangents
        output_cotangents = []
        for out_impl in alive_outputs:
            if out_impl is not None:
                if id(out_impl) in cotangent_map:
                    cot_impl = cotangent_map[id(out_impl)]
                    # print(f"  [BACKPROP] Cot for output {id(out_impl)}: shards={len(cot_impl._values)}, sharding={cot_impl.sharding}")
                    output_cotangents.append(Tensor(impl=cot_impl))
                else:
                    # If this specific output wasn't used, use zeros
                    from ...ops.creation import zeros_like

                    output_cotangents.append(zeros_like(Tensor(impl=out_impl)))
            else:
                output_cotangents.append(None)

        cotangents_structured = pytree.tree_unflatten(
            output_refs.tree_def, output_cotangents
        )

        # 4. Invoke VJP Rule with automatic unwrapping
        vjp_primals = _unwrap_single(primals_structured)
        vjp_cotangents = _unwrap_single(cotangents_structured)
        vjp_outputs = _unwrap_single(outputs_structured)

        # Apply trace state suppression if needed
        all_vjp_inputs = pytree.tree_leaves([vjp_primals, vjp_cotangents, vjp_outputs])
        for x in all_vjp_inputs:
            set_trace_state(x)

        # print(f"  [BACKPROP] Op: {op.name}")
        try:
            input_cotangents = op.vjp_rule(vjp_primals, vjp_cotangents, vjp_outputs)
        except Exception as e:
            # Restore state before raising
            for x in all_vjp_inputs:
                restore_trace_state(x)
            op_name = getattr(op, "name", str(op))
            raise RuntimeError(f"VJP rule failed for operation '{op_name}': {e}") from e

        # Restore state immediately after VJP execution
        for x in all_vjp_inputs:
            restore_trace_state(x)

        # 5. Accumulate Cotangents back to inputs
        arg_impls = [
            arg if isinstance(arg, (TensorImpl, Tensor)) else None
            for arg in pytree.tree_leaves(output_refs.op_args)
        ]

        # Wrap result in tuple if it was a single tensor for multiple args (unlikely but safe)
        if not isinstance(input_cotangents, (list, tuple)) and len(arg_impls) == 1:
            input_cotangents = (input_cotangents,)

        cotangent_result_leaves = pytree.tree_leaves(input_cotangents)

        for arg, cot_result in zip(arg_impls, cotangent_result_leaves, strict=False):
            if arg is None or cot_result is None:
                continue

            arg_impl = arg._impl if isinstance(arg, Tensor) else arg
            arg_id = id(arg_impl)

            # Ensure cot_result is a Tensor for accumulation
            if isinstance(cot_result, TensorImpl):
                cot_tensor = Tensor(impl=cot_result)
            elif isinstance(cot_result, Tensor):
                cot_tensor = cot_result
            else:
                # Should be a Tensor/TensorImpl already if VJP rule is correct
                continue

            arg_shape = Tensor(impl=arg_impl).shape
            if len(cot_tensor.shape) > len(arg_shape):
                diff = len(cot_tensor.shape) - len(arg_shape)
                # Assume extra dims are at the front (standard broadcasting/vmap rules)
                reduce_axes = list(range(diff))
                from ...ops.reduction import reduce_sum

                cot_tensor = reduce_sum(cot_tensor, axis=reduce_axes)

            # CRITICAL: Preserve sharding of the original argument
            from ...ops.communication import reshard
            from ...core.sharding.spec import needs_reshard, ShardingSpec

            if arg_impl.sharding and needs_reshard(
                cot_tensor.sharding, arg_impl.sharding
            ):
                cot_tensor = reshard(
                    cot_tensor,
                    arg_impl.sharding.mesh,
                    arg_impl.sharding.dim_specs,
                    replicated_axes=arg_impl.sharding.replicated_axes,
                )

            if arg_id in cotangent_map:
                from ...ops.binary import add

                existing = Tensor(impl=cotangent_map[arg_id])

                accumulated = add(existing, cot_tensor)

                cotangent_map[arg_id] = accumulated._impl
            else:
                cotangent_map[arg_id] = cot_tensor._impl

    # Construct final gradient mapping for trace inputs
    gradients = {}
    from ...ops.communication import reshard, all_reduce
    from ...core.sharding.spec import needs_reshard

    for inp in input_leaves:
        inp_id = id(inp._impl)
        if inp_id in cotangent_map:
            grad = Tensor(impl=cotangent_map[inp_id])

            # Resolve Partial Sums if any
            if grad.sharding and grad.sharding.partial_sum_axes:
                grad = all_reduce(
                    grad, reduce_axes=list(grad.sharding.partial_sum_axes)
                )

            # Ensure grad has same sharding as input to be future-proof (ZeRO/Pipeline)
            if inp.sharding and needs_reshard(grad.sharding, inp.sharding):
                grad = reshard(
                    grad,
                    inp.sharding.mesh,
                    inp.sharding.dim_specs,
                    replicated_axes=inp.sharding.replicated_axes,
                )

            gradients[inp] = grad
        else:
            from ...ops.creation import zeros_like, full

            # Zero gradient must handle sharding too!
            # zeros_like(inp) inherits sharding logic usually?
            # If inp is sharded, zeros_like should be sharded.
            z = zeros_like(inp)
            gradients[inp] = z

    return gradients


__all__ = ["backward_on_trace"]
